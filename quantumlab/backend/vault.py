"""
Secure vault for managing SSH credentials and secrets.
Uses Fernet symmetric encryption for credential storage.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import sqlite3
from contextlib import contextmanager

VAULT_DB = Path(__file__).parent / "vault.db"
# In production, this should be from environment variable or HSM
VAULT_MASTER_KEY = os.getenv("VAULT_MASTER_KEY", "default-dev-key-change-in-production")


class SecureVault:
    """Manages encrypted storage of credentials and secrets."""
    
    def __init__(self, db_path: Path = VAULT_DB):
        self.db_path = db_path
        self.cipher = self._get_cipher()
        self.init_database()
    
    def _get_cipher(self):
        """Initialize Fernet cipher with derived key."""
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'quantumlab-vault-salt',  # In production, use random salt per installation
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(VAULT_MASTER_KEY.encode()))
        return Fernet(key)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize vault database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # SSH credentials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ssh_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER DEFAULT 22,
                    username TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    encrypted_password TEXT,
                    encrypted_private_key TEXT,
                    encrypted_passphrase TEXT,
                    fingerprint TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    tags TEXT
                )
            """)
            
            # API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    encrypted_secret TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                )
            """)
            
            # VM instances table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vm_instances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    provider TEXT NOT NULL,
                    instance_id TEXT,
                    region TEXT,
                    instance_type TEXT,
                    status TEXT DEFAULT 'stopped',
                    public_ip TEXT,
                    private_ip TEXT,
                    ssh_credential_id INTEGER,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (ssh_credential_id) REFERENCES ssh_credentials(id)
                )
            """)
            
            # Training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    vm_instance_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_path TEXT,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    start_time TEXT,
                    end_time TEXT,
                    logs TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (vm_instance_id) REFERENCES vm_instances(id)
                )
            """)
            
            # Audit log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id INTEGER,
                    user TEXT,
                    ip_address TEXT,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vm_status ON vm_instances(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training_sessions(status)")
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def save_ssh_credential(
        self,
        name: str,
        host: str,
        username: str,
        auth_type: str,  # 'password' or 'key'
        password: Optional[str] = None,
        private_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        port: int = 22,
        tags: Optional[List[str]] = None
    ) -> int:
        """Save SSH credentials securely."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            # Encrypt sensitive data
            encrypted_password = self.encrypt(password) if password else None
            encrypted_key = self.encrypt(private_key) if private_key else None
            encrypted_passphrase = self.encrypt(passphrase) if passphrase else None
            
            cursor.execute("""
                INSERT INTO ssh_credentials 
                (name, host, port, username, auth_type, encrypted_password, 
                 encrypted_private_key, encrypted_passphrase, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, host, port, username, auth_type,
                encrypted_password, encrypted_key, encrypted_passphrase,
                now, now, json.dumps(tags) if tags else None
            ))
            
            cred_id = cursor.lastrowid
            
            # Audit log
            cursor.execute("""
                INSERT INTO audit_log (action, resource_type, resource_id, timestamp, details)
                VALUES (?, ?, ?, ?, ?)
            """, ('create', 'ssh_credential', cred_id, now, json.dumps({'name': name, 'host': host})))
            
            return cred_id
    
    def get_ssh_credential(self, credential_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt SSH credentials."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ssh_credentials WHERE id = ?", (credential_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            cred = dict(row)
            
            # Decrypt sensitive fields
            if cred['encrypted_password']:
                cred['password'] = self.decrypt(cred['encrypted_password'])
            if cred['encrypted_private_key']:
                cred['private_key'] = self.decrypt(cred['encrypted_private_key'])
            if cred['encrypted_passphrase']:
                cred['passphrase'] = self.decrypt(cred['encrypted_passphrase'])
            
            # Remove encrypted fields from response
            cred.pop('encrypted_password', None)
            cred.pop('encrypted_private_key', None)
            cred.pop('encrypted_passphrase', None)
            
            # Update last used
            now = datetime.now().isoformat()
            cursor.execute("UPDATE ssh_credentials SET last_used_at = ? WHERE id = ?", (now, credential_id))
            
            return cred
    
    def list_ssh_credentials(self) -> List[Dict[str, Any]]:
        """List all SSH credentials (without decrypting sensitive data)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, host, port, username, auth_type, created_at, last_used_at, tags
                FROM ssh_credentials
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_ssh_credential(self, credential_id: int):
        """Delete SSH credentials."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ssh_credentials WHERE id = ?", (credential_id,))
            
            # Audit log
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO audit_log (action, resource_type, resource_id, timestamp)
                VALUES (?, ?, ?, ?)
            """, ('delete', 'ssh_credential', credential_id, now))
    
    def save_api_key(self, name: str, provider: str, key: str, secret: Optional[str] = None) -> int:
        """Save API key securely."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            encrypted_key = self.encrypt(key)
            encrypted_secret = self.encrypt(secret) if secret else None
            
            cursor.execute("""
                INSERT INTO api_keys (name, provider, encrypted_key, encrypted_secret, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, provider, encrypted_key, encrypted_secret, now, now))
            
            return cursor.lastrowid
    
    def get_api_key(self, key_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt API key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            key_data = dict(row)
            key_data['key'] = self.decrypt(key_data['encrypted_key'])
            if key_data['encrypted_secret']:
                key_data['secret'] = self.decrypt(key_data['encrypted_secret'])
            
            key_data.pop('encrypted_key', None)
            key_data.pop('encrypted_secret', None)
            
            # Update last used
            now = datetime.now().isoformat()
            cursor.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id))
            
            return key_data
    
    def register_vm(
        self,
        name: str,
        provider: str,
        instance_type: str,
        ssh_credential_id: Optional[int] = None,
        instance_id: Optional[str] = None,
        region: Optional[str] = None,
        public_ip: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Register a VM instance."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO vm_instances 
                (name, provider, instance_id, region, instance_type, ssh_credential_id, 
                 public_ip, metadata, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, provider, instance_id, region, instance_type, ssh_credential_id,
                public_ip, json.dumps(metadata) if metadata else None, 'stopped', now, now
            ))
            
            return cursor.lastrowid
    
    def update_vm_status(self, vm_id: int, status: str, public_ip: Optional[str] = None):
        """Update VM status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            if public_ip:
                cursor.execute("""
                    UPDATE vm_instances 
                    SET status = ?, public_ip = ?, updated_at = ?
                    WHERE id = ?
                """, (status, public_ip, now, vm_id))
            else:
                cursor.execute("""
                    UPDATE vm_instances 
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                """, (status, now, vm_id))
    
    def get_vm(self, vm_id: int) -> Optional[Dict[str, Any]]:
        """Get VM instance details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM vm_instances WHERE id = ?", (vm_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_vms(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List VM instances."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("SELECT * FROM vm_instances WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM vm_instances ORDER BY created_at DESC")
            
            return [dict(row) for row in cursor.fetchall()]
    
    def create_training_session(
        self,
        name: str,
        vm_instance_id: int,
        model_type: str,
        dataset_path: Optional[str] = None
    ) -> int:
        """Create a new training session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            
            cursor.execute("""
                INSERT INTO training_sessions 
                (name, vm_instance_id, model_type, dataset_path, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, vm_instance_id, model_type, dataset_path, 'pending', now))
            
            return cursor.lastrowid
    
    def update_training_session(
        self,
        session_id: int,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        logs: Optional[str] = None
    ):
        """Update training session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
                if status == 'running' and not self.get_training_session(session_id).get('start_time'):
                    updates.append("start_time = ?")
                    params.append(datetime.now().isoformat())
                elif status in ['completed', 'failed']:
                    updates.append("end_time = ?")
                    params.append(datetime.now().isoformat())
            
            if progress is not None:
                updates.append("progress = ?")
                params.append(progress)
            
            if logs:
                updates.append("logs = ?")
                params.append(logs)
            
            if updates:
                params.append(session_id)
                cursor.execute(f"UPDATE training_sessions SET {', '.join(updates)} WHERE id = ?", params)
    
    def get_training_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get training session details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_training_sessions(self, vm_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List training sessions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if vm_id:
                cursor.execute("""
                    SELECT * FROM training_sessions 
                    WHERE vm_instance_id = ? 
                    ORDER BY created_at DESC
                """, (vm_id,))
            else:
                cursor.execute("SELECT * FROM training_sessions ORDER BY created_at DESC")
            
            return [dict(row) for row in cursor.fetchall()]


# Global vault instance
vault = SecureVault()
