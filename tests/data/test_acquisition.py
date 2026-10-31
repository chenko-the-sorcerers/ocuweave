"""Reject corrupt data without replacing an existing archive."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ocuweave.data.acquisition import download_verified


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        disk=patch('ocuweave.data.acquisition.shutil.disk_usage',return_value=SimpleNamespace(free=1024**3))
        disk.start();self.addCleanup(disk.stop)

    def test_insufficient_space_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('ocuweave.data.acquisition.shutil.disk_usage',return_value=SimpleNamespace(free=0)), patch('urllib.request.urlopen',side_effect=AssertionError('network must not be used')):
                with self.assertRaises(OSError):
                    download_verified('https://example.org/data.zip',Path(directory)/'data.zip',3,'sha256','0'*64)

    def test_verification_reuse_and_preservation(self):
        data=b'official fixture';digest=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.zip'
            with patch('urllib.request.urlopen',return_value=io.BytesIO(data)):
                self.assertEqual(download_verified('https://example.org/data.zip',path,len(data),'sha256',digest),path)
            with patch('urllib.request.urlopen',side_effect=AssertionError('must reuse')):
                download_verified('https://example.org/data.zip',path,len(data),'sha256',digest)
            with self.assertRaises(ValueError):
                download_verified('https://example.org/data.zip',path,len(data),'sha256','0'*64)
            self.assertEqual(path.read_bytes(),data)

    def test_failed_download_never_becomes_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.zip'
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):
                    download_verified('https://example.org/data.zip',path,3,'sha256','0'*64)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.zip.part').exists())
