"""TLS certificate generation for MITM proxy."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

STARNOSE_DIR = Path("~/.starnose").expanduser()
CA_CERT_PATH = STARNOSE_DIR / "ca.pem"
CA_KEY_PATH = STARNOSE_DIR / "ca-key.pem"
CERTS_DIR = STARNOSE_DIR / "certs"


def get_or_create_ca() -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Get existing CA or generate a new one."""
    STARNOSE_DIR.mkdir(parents=True, exist_ok=True)

    if CA_CERT_PATH.exists() and CA_KEY_PATH.exists():
        ca_key = serialization.load_pem_private_key(
            CA_KEY_PATH.read_bytes(), password=None
        )
        ca_cert = x509.load_pem_x509_certificate(CA_CERT_PATH.read_bytes())
        return ca_cert, ca_key

    # Generate CA key
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Generate self-signed CA cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "starnose CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "starnose"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    CA_KEY_PATH.write_bytes(
        ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    CA_CERT_PATH.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    os.chmod(CA_KEY_PATH, 0o600)

    return ca_cert, ca_key


def create_server_cert(hostname: str) -> tuple[Path, Path]:
    """Create a server certificate for a hostname, signed by the starnose CA."""
    CERTS_DIR.mkdir(parents=True, exist_ok=True)

    cert_path = CERTS_DIR / f"{hostname}.pem"
    key_path = CERTS_DIR / f"{hostname}-key.pem"

    # Return cached cert if still valid
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if cert.not_valid_after_utc > datetime.now(timezone.utc):
                return cert_path, key_path
        except Exception:
            pass

    ca_cert, ca_key = get_or_create_ca()

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
    ])

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    # Write server cert + CA cert chain so clients can verify
    cert_pem = server_cert.public_bytes(serialization.Encoding.PEM)
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(cert_pem + ca_pem)

    os.chmod(key_path, 0o600)

    return cert_path, key_path
