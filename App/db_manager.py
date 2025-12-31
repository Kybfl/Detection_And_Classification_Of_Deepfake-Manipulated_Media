import sqlite3
import hashlib
import datetime
import os

class AuditLogger:
    def __init__(self, db_name="logs.db"):
        """
        Initializes the database connection string.
        """
        # db_manager path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # full db path
        self.db_path = os.path.join(base_dir, db_name)
        
        self.db_name = db_name
        self._initialize_tables()

    def _initialize_tables(self):
        """
        Creates the necessary tables if they don't exist.
        Private method, should not be called from outside.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS scan_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    file_hash TEXT,
                    file_type TEXT,
                    prediction_result TEXT,
                    confidence_score REAL,
                    status TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database Initialization Error: {e}")

    def _calculate_file_hash(self, file_bytes): #hashing the file to prevent any legal or illegal accusations
        """
        Generates SHA-256 hash of the file for integrity verification.
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_bytes)
        return sha256_hash.hexdigest()

    def log_transaction(self, filename, file_bytes, file_type, result, confidence):
        """
        Logs the transaction details into the database.
        
        Args:
            filename (str): Name of the uploaded file
            file_bytes (bytes): Raw content of the file (for hashing)
            file_type (str): 'Image' or 'Video'
            result (str): The prediction class
            confidence (float): The confidence score 
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            file_hash = self._calculate_file_hash(file_bytes)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute('''
                INSERT INTO scan_logs (timestamp, filename, file_hash, file_type, prediction_result, confidence_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, filename, file_hash, file_type, result, confidence, "SUCCESS"))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Logging Error: {e}")
            return False