import pytest
import torch
from unittest.mock import patch, MagicMock
from crucible.api.model_server import load_model, generate_response


class TestModelServer:
    def test_model_loading_error_handling(self):
        """Test that model loading errors are handled gracefully"""
        with patch('crucible.api.model_server.AutoTokenizer') as mock_tokenizer, \
             patch('crucible.api.model_server.AutoModelForCausalLM') as mock_model:
            
            # Simulate loading failure
            mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
            
            with pytest.raises(RuntimeError, match="Failed to load model"):
                load_model("nonexistent_model")

    def test_model_caching(self):
        """Test that models are properly cached"""
        with patch('crucible.api.model_server.AutoTokenizer') as mock_tokenizer, \
             patch('crucible.api.model_server.AutoModelForCausalLM') as mock_model:
            
            # Mock the model and tokenizer
            mock_model_instance = MagicMock()
            mock_tokenizer_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Load model twice
            model1, tokenizer1 = load_model("test_model")
            model2, tokenizer2 = load_model("test_model")
            
            # Should only be called once due to caching
            assert mock_model.from_pretrained.call_count == 1
            assert mock_tokenizer.from_pretrained.call_count == 1
            assert model1 is model2
            assert tokenizer1 is tokenizer2

    def test_generate_response_error_handling(self):
        """Test that generation errors are handled"""
        with patch('crucible.api.model_server.load_model') as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Simulate generation failure
            mock_model.generate.side_effect = Exception("Generation failed")
            
            with pytest.raises(Exception):
                generate_response("test input")

    def test_model_eval_mode(self):
        """Test that model is set to evaluation mode"""
        with patch('crucible.api.model_server.AutoTokenizer') as mock_tokenizer, \
             patch('crucible.api.model_server.AutoModelForCausalLM') as mock_model:
            
            mock_model_instance = MagicMock()
            mock_tokenizer_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Clear any existing cache
            from crucible.api.model_server import MODEL_CACHE
            MODEL_CACHE.clear()
            
            load_model("test_model")
            
            # Verify eval() was called
            mock_model_instance.eval.assert_called_once()

    def test_generate_response_output_processing(self):
        """Test that output is properly processed"""
        with patch('crucible.api.model_server.load_model') as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Mock the generation process
            mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
            mock_tokenizer.decode.return_value = "input text generated text"
            mock_tokenizer.eos_token_id = 0
            
            # Mock model generation
            mock_output_ids = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model.generate.return_value = mock_output_ids
            
            result = generate_response("input text")
            
            # Should return only the generated part
            assert result == "generated text"
            mock_tokenizer.encode.assert_called_once_with("input text", return_tensors="pt")
            mock_model.generate.assert_called_once()
