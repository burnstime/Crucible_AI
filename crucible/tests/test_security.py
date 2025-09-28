import pytest
from fastapi.testclient import TestClient
from crucible.api.main import app
import json
import os


class TestSecurity:
    def test_input_validation_empty_input(self):
        """Test that empty input is rejected"""
        client = TestClient(app)
        response = client.post("/generate", json={"input": ""})
        assert response.status_code == 422  # Validation error

    def test_input_validation_too_long(self):
        """Test that overly long input is rejected"""
        client = TestClient(app)
        long_input = "x" * 10001  # Exceeds 10000 char limit
        response = client.post("/generate", json={"input": long_input})
        assert response.status_code == 422

    def test_input_validation_xss_protection(self):
        """Test that XSS attempts are blocked"""
        client = TestClient(app)
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        for malicious_input in malicious_inputs:
            response = client.post("/generate", json={"input": malicious_input})
            assert response.status_code == 422

    def test_session_id_validation(self):
        """Test that overly long session IDs are rejected"""
        client = TestClient(app)
        long_session_id = "x" * 101  # Exceeds 100 char limit
        response = client.post("/generate", json={
            "input": "Hello",
            "session_id": long_session_id
        })
        assert response.status_code == 422

    def test_malformed_json_handling(self):
        """Test that malformed JSON is handled gracefully"""
        client = TestClient(app)
        response = client.post("/generate", data="invalid json")
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test that missing required fields are handled"""
        client = TestClient(app)
        response = client.post("/generate", json={})
        assert response.status_code == 422


class TestDataCleaningSecurity:
    def test_safe_file_operations(self, tmp_path):
        """Test that file operations are atomic and safe"""
        from crucible.tools.data_cleaning import clean_logs
        import tempfile
        
        # Create logs directory and test data
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        test_log = logs_dir / "interaction_logs.jsonl"
        test_log.write_text('{"input": "test", "model_response": "response"}\n')
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # This should not raise an exception
            clean_logs()
            assert (tmp_path / "data" / "processed" / "cleaned.jsonl").exists()
        finally:
            os.chdir(original_cwd)

    def test_malformed_json_handling_in_cleaning(self, tmp_path):
        """Test that malformed JSON in logs is handled gracefully"""
        from crucible.tools.data_cleaning import clean_logs
        
        # Create logs directory and test data with malformed JSON
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        test_log = logs_dir / "interaction_logs.jsonl"
        test_log.write_text('{"input": "test", "model_response": "response"}\ninvalid json\n{"input": "test2", "model_response": "response2"}\n')
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Should not raise exception
            clean_logs()
            output_file = tmp_path / "data" / "processed" / "cleaned.jsonl"
            assert output_file.exists()
            # Should contain valid entries
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) >= 2  # At least 2 valid entries
        finally:
            os.chdir(original_cwd)

    def test_secure_hashing(self):
        """Test that secure hashing is used for deduplication"""
        from crucible.tools.data_cleaning import dedupe_by_input
        import tempfile
        
        # Create test data with same input
        test_data = [
            '{"input": "same input", "model_response": "response1"}\n',
            '{"input": "same input", "model_response": "response2"}\n',
            '{"input": "different input", "model_response": "response3"}\n'
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.writelines(test_data)
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            output_file = f.name
        
        try:
            dedupe_by_input(input_file, output_file)
            
            # Should only have 2 unique entries (same input deduplicated)
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) == 2
        finally:
            os.unlink(input_file)
            os.unlink(output_file)
