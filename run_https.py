"""Run the Streamlit app over HTTPS for LAN camera access.

The generated certificate is self-signed and intended for local development.
Browsers may show a certificate warning the first time; camera permissions are
available after the user proceeds to the HTTPS page.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


PROJECT_DIR = Path(__file__).resolve().parent
CERT_DIR = PROJECT_DIR / ".streamlit" / "certs"
CERT_PATH = CERT_DIR / "age_gender_local.crt"
KEY_PATH = CERT_DIR / "age_gender_local.key"


def _local_ipv4_addresses() -> set[str]:
    addresses = {"127.0.0.1"}
    try:
        _, _, resolved = socket.gethostbyname_ex(socket.gethostname())
        for value in resolved:
            try:
                parsed = ipaddress.ip_address(value)
            except ValueError:
                continue
            if parsed.version == 4:
                addresses.add(str(parsed))
    except OSError:
        pass
    return addresses


def _certificate_is_current() -> bool:
    if not CERT_PATH.exists() or not KEY_PATH.exists():
        return False
    try:
        certificate = x509.load_pem_x509_certificate(CERT_PATH.read_bytes())
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        names = set(san.get_values_for_type(x509.DNSName))
        names.update(str(value) for value in san.get_values_for_type(x509.IPAddress))
        expected_names = {"localhost"} | _local_ipv4_addresses()
        expiry = getattr(certificate, "not_valid_after_utc", None)
        if expiry is None:
            expiry = certificate.not_valid_after.replace(tzinfo=timezone.utc)
        return expected_names.issubset(names) and expiry > datetime.now(timezone.utc) + timedelta(days=30)
    except Exception:
        return False


def _create_certificate() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Age & Gender AI local HTTPS")]
    )
    names = [x509.DNSName("localhost")]
    names.extend(x509.IPAddress(ipaddress.ip_address(value)) for value in _local_ipv4_addresses())
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=5))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_PATH.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    CERT_PATH.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def main() -> int:
    if not _certificate_is_current():
        _create_certificate()
        print(f"Created local certificate: {CERT_PATH}")

    environment = os.environ.copy()
    environment["AGE_GENDER_HTTPS"] = "1"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_DIR / "app.py"),
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
        f"--server.sslCertFile={CERT_PATH}",
        f"--server.sslKeyFile={KEY_PATH}",
    ]
    print("Open https://localhost:8501 locally or https://<this-computer-IP>:8501 on your LAN.")
    return subprocess.call(command, cwd=PROJECT_DIR, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
