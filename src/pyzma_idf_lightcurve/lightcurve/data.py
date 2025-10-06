"""
Data structures and database schemas for lightcurve analysis.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass


@dataclass
class ObjectMetadata:
    """Metadata for each detected object from superstack."""
    object_id: int              # Unique object ID
    parent_id: Optional[int]    # Parent ID for deblending groups (None if not deblended)
    ra: float                   # Right Ascension (degrees)
    dec: float                  # Declination (degrees)  
    x_superstack: float         # X pixel coordinate in superstack
    y_superstack: float         # Y pixel coordinate in superstack
    mag_auto_superstack: float  # Reference magnitude from superstack
    flux_auto_superstack: float # Reference flux from superstack
    class_star: float           # Star/galaxy classifier
    flags: int                  # SExtractor flags
    fwhm_image: float          # FWHM in pixels
    ellipticity: float         # Object ellipticity
    kron_radius: float         # Kron radius
    channel: Literal["ch1", "ch2"]  # Which channel this object was detected in

@dataclass  
class AORMetadata:
    """Metadata for each AOR observation."""
    aor_number: str            # AOR identifier (e.g., "r58520832")
    obs_time: float           # Observation time (MJD)
    exptime: float            # Exposure time (seconds)
    group_name: str           # Group name (e.g., "gr123")
    channel: Literal["ch1", "ch2"]
    obs_date: str             # Human readable date
    program_id: str           # Spitzer program ID
    
@dataclass
class FluxMeasurement:
    """Single flux measurement for one object in one AOR."""
    object_id: int
    aor_number: str
    measurement_method: Literal["original_sci", "lac_cleaned_sci", "division_method"]
    flux_type: str            # "mag_auto", "mag_aper_1", "mag_aper_2", etc.
    flux: float              # Measured flux 
    flux_err: float          # Flux uncertainty
    magnitude: float         # Magnitude (if applicable)
    mag_err: float           # Magnitude uncertainty
    flags: int               # Measurement flags
    background: float        # Local background estimate
    x_image: float           # X coordinate in this image
    y_image: float           # Y coordinate in this image


class LightcurveDatabase:
    """
    Optimized database design for lightcurve storage and querying.
    
    Uses multiple tables with proper indexing for high-performance queries.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_tables()
        self.create_indexes()
    
    def setup_tables(self):
        """Create optimized table structure."""
        
        # Objects table - one row per detected object
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            object_id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            ra REAL NOT NULL,
            dec REAL NOT NULL,
            x_superstack REAL,
            y_superstack REAL,
            mag_auto_superstack REAL,
            flux_auto_superstack REAL,
            class_star REAL,
            flags INTEGER,
            fwhm_image REAL,
            ellipticity REAL,
            kron_radius REAL,
            channel TEXT NOT NULL CHECK(channel IN ('ch1', 'ch2'))
        )
        """)
        
        # AORs table - one row per AOR
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS aors (
            aor_number TEXT PRIMARY KEY,
            obs_time REAL NOT NULL,  -- MJD
            exptime REAL,
            group_name TEXT,
            channel TEXT NOT NULL CHECK(channel IN ('ch1', 'ch2')),
            obs_date TEXT,
            program_id TEXT
        )
        """)
        
        # Measurements table - one row per flux measurement
        # This is the main lightcurve data table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER NOT NULL,
            aor_number TEXT NOT NULL,
            measurement_method TEXT NOT NULL CHECK(
                measurement_method IN ('original_sci', 'lac_cleaned_sci', 'division_method')
            ),
            flux_type TEXT NOT NULL,  -- 'mag_auto', 'mag_aper_1', etc.
            flux REAL,
            flux_err REAL,
            magnitude REAL,
            mag_err REAL,
            flags INTEGER,
            background REAL,
            x_image REAL,
            y_image REAL,
            FOREIGN KEY (object_id) REFERENCES objects (object_id),
            FOREIGN KEY (aor_number) REFERENCES aors (aor_number)
        )
        """)
        
    def create_indexes(self):
        """Create indexes for fast queries."""
        
        # Primary access patterns
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_object_id ON measurements (object_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_aor_number ON measurements (aor_number)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_measurement_method ON measurements (measurement_method)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_flux_type ON measurements (flux_type)")
        
        # Composite indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_object_method_flux ON measurements (object_id, measurement_method, flux_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_aor_time ON aors (obs_time)")
        
        # Spatial indexes for objects
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ra_dec ON objects (ra, dec)")
        
        self.conn.commit()


class LightcurveQueryEngine:
    """High-performance query engine for lightcurve data."""
    
    def __init__(self, database: LightcurveDatabase):
        self.db = database
        self.conn = database.conn
    
    def get_lightcurve(
        self,
        object_id: int,
        measurement_method: str = "original_sci",
        flux_type: str = "mag_auto",
        time_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Get lightcurve for a specific object.
        
        Args:
            object_id: Object identifier
            measurement_method: Method used for measurements
            flux_type: Type of flux measurement
            time_range: Optional (start_mjd, end_mjd) tuple
            
        Returns:
            DataFrame with time, magnitude, error, flags
        """
        
        query = """
        SELECT a.obs_time, m.magnitude, m.mag_err, m.flags, a.aor_number, a.group_name
        FROM measurements m
        JOIN aors a ON m.aor_number = a.aor_number
        WHERE m.object_id = ? AND m.measurement_method = ? AND m.flux_type = ?
        """
        
        params = [object_id, measurement_method, flux_type]
        
        if time_range:
            query += " AND a.obs_time BETWEEN ? AND ?"
            params.extend(time_range)
            
        query += " ORDER BY a.obs_time"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_object_metadata(self, object_id: int) -> Optional[Dict]:
        """Get metadata for a specific object."""
        
        query = "SELECT * FROM objects WHERE object_id = ?"
        cursor = self.conn.execute(query, [object_id])
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def search_variable_objects(
        self,
        min_observations: int = 10,
        max_magnitude: float = 20.0,
        measurement_method: str = "original_sci",
        flux_type: str = "mag_auto"
    ) -> pd.DataFrame:
        """
        Find potentially variable objects based on magnitude scatter.
        
        Args:
            min_observations: Minimum number of observations required
            max_magnitude: Maximum mean magnitude  
            measurement_method: Method to use for analysis
            flux_type: Flux type to analyze
            
        Returns:
            DataFrame with object stats and variability metrics
        """
        
        query = """
        SELECT 
            m.object_id,
            o.ra, o.dec, o.mag_auto_superstack,
            COUNT(*) as n_obs,
            AVG(m.magnitude) as mean_mag,
            STDEV(m.magnitude) as mag_std,
            MIN(m.magnitude) as min_mag,
            MAX(m.magnitude) as max_mag,
            (MAX(m.magnitude) - MIN(m.magnitude)) as mag_range
        FROM measurements m
        JOIN objects o ON m.object_id = o.object_id
        WHERE m.measurement_method = ? AND m.flux_type = ?
        GROUP BY m.object_id, o.ra, o.dec, o.mag_auto_superstack
        HAVING COUNT(*) >= ? AND AVG(m.magnitude) <= ?
        ORDER BY mag_std DESC
        """
        
        return pd.read_sql_query(
            query, self.conn, 
            params=[measurement_method, flux_type, min_observations, max_magnitude]
        )