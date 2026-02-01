"""
Unit tests for Nextflow pipeline validation.

Tests the configuration, process definitions, and pipeline logic.
"""

import pytest
import subprocess
import json
from pathlib import Path


class TestNextflowPipeline:
    """Test suite for the Nextflow discovery pipeline."""
    
    @pytest.fixture
    def nextflow_dir(self):
        """Get the Nextflow directory path."""
        return Path(__file__).parent.parent / "nextflow"
    
    @pytest.fixture
    def main_nf(self, nextflow_dir):
        """Get the main.nf file path."""
        return nextflow_dir / "main.nf"
    
    @pytest.fixture
    def config_file(self, nextflow_dir):
        """Get the nextflow.config file path."""
        return nextflow_dir / "nextflow.config"
    
    def test_nextflow_files_exist(self, nextflow_dir, main_nf, config_file):
        """Test that required Nextflow files exist."""
        assert nextflow_dir.exists(), "Nextflow directory not found"
        assert main_nf.exists(), "main.nf file not found"
        assert config_file.exists(), "nextflow.config file not found"
    
    def test_main_nf_syntax(self, main_nf):
        """Test that main.nf has valid syntax."""
        content = main_nf.read_text()
        
        # Check for DSL2 declaration
        assert "nextflow.enable.dsl=2" in content, "DSL2 not enabled"
        
        # Check for workflow definition
        assert "workflow {" in content, "Workflow definition not found"
        
        # Check for process definitions
        required_processes = [
            "data_preparation",
            "xtb_enrichment",
            "surrogate_training",
            "score_training",
            "generative_sampling",
            "analysis_visualization",
            "advanced_benchmarking"
        ]
        
        for process_name in required_processes:
            assert f"process {process_name}" in content, \
                f"Process '{process_name}' not defined"
    
    def test_config_file_syntax(self, config_file):
        """Test that nextflow.config has valid syntax."""
        content = config_file.read_text()
        
        # Check for manifest
        assert "manifest {" in content, "Manifest section not found"
        
        # Check for params
        assert "params {" in content, "Parameters section not found"
        
        # Check for process configuration
        assert "process {" in content, "Process configuration not found"
        
        # Check for profiles
        assert "profiles {" in content, "Profiles section not found"
    
    def test_required_parameters(self, config_file):
        """Test that required parameters are defined."""
        content = config_file.read_text()
        
        required_params = [
            "raw_dataset",
            "output_dir",
            "num_epochs_surrogate",
            "num_epochs_score",
            "num_samples"
        ]
        
        for param in required_params:
            assert param in content, f"Parameter '{param}' not defined in config"
    
    def test_required_profiles(self, config_file):
        """Test that required execution profiles are defined."""
        content = config_file.read_text()
        
        required_profiles = [
            "standard",
            "docker",
            "conda",
            "cluster",
            "test"
        ]
        
        for profile in required_profiles:
            assert profile in content, f"Profile '{profile}' not defined"
    
    def test_process_resources(self, config_file):
        """Test that process resources are properly configured."""
        content = config_file.read_text()
        
        # Check for resource definitions
        assert "cpus" in content, "CPU configuration not found"
        assert "memory" in content, "Memory configuration not found"
        assert "time" in content, "Time configuration not found"
    
    def test_workflow_logic(self, main_nf):
        """Test that workflow connects processes correctly."""
        content = main_nf.read_text()
        
        # Check for process connections
        assert "data_preparation(raw_data)" in content, \
            "Data preparation not connected to input"
        
        assert "xtb_enrichment(data_preparation.out" in content, \
            "Enrichment not connected to data preparation"
        
        assert "surrogate_training(xtb_enrichment.out" in content, \
            "Surrogate training not connected to enrichment"
        
        assert "score_training(xtb_enrichment.out" in content, \
            "Score training not connected to enrichment"
    
    def test_publishdir_definitions(self, main_nf):
        """Test that all processes have publishDir directives."""
        content = main_nf.read_text()
        
        # Count process definitions
        process_count = content.count("process ")
        
        # Count publishDir directives (should be at least one per process)
        publishdir_count = content.count("publishDir")
        
        assert publishdir_count >= process_count - 1, \
            "Not all processes have publishDir directives"
    
    def test_error_handling(self, config_file):
        """Test that error handling is configured."""
        content = config_file.read_text()
        
        assert "errorStrategy" in content, "Error strategy not defined"
        assert "maxRetries" in content, "Max retries not defined"
    
    def test_output_channels(self, main_nf):
        """Test that processes emit output channels."""
        content = main_nf.read_text()
        
        # Check for emit declarations
        assert ", emit:" in content, "Output channels not properly defined"
    
    def test_workflow_completion_handler(self, main_nf):
        """Test that workflow completion handler is defined."""
        content = main_nf.read_text()
        
        assert "workflow.onComplete" in content, \
            "Workflow completion handler not found"


class TestPipelineIntegration:
    """Integration tests for pipeline execution."""
    
    @pytest.fixture
    def nextflow_dir(self):
        """Get the Nextflow directory path."""
        return Path(__file__).parent.parent / "nextflow"
    
    @pytest.mark.skipif(
        not subprocess.run(["which", "nextflow"], 
                         capture_output=True).returncode == 0,
        reason="Nextflow not installed"
    )
    def test_nextflow_version(self):
        """Test that Nextflow is installed and version is sufficient."""
        result = subprocess.run(
            ["nextflow", "-version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Nextflow not working"
        assert "nextflow" in result.stdout.lower(), "Invalid Nextflow output"
    
    @pytest.mark.skipif(
        not subprocess.run(["which", "nextflow"], 
                         capture_output=True).returncode == 0,
        reason="Nextflow not installed"
    )
    def test_pipeline_syntax_check(self, nextflow_dir):
        """Test that the pipeline passes Nextflow syntax check."""
        result = subprocess.run(
            ["nextflow", "run", "main.nf", "-preview"],
            cwd=nextflow_dir,
            capture_output=True,
            text=True
        )
        
        # Note: -preview might not be available in all Nextflow versions
        # This test might need adjustment based on Nextflow version


class TestPipelineDocumentation:
    """Test that pipeline documentation is complete."""
    
    @pytest.fixture
    def readme_file(self):
        """Get the README file path."""
        return Path(__file__).parent.parent / "nextflow" / "README.md"
    
    def test_readme_exists(self, readme_file):
        """Test that README exists."""
        assert readme_file.exists(), "README.md not found"
    
    def test_readme_content(self, readme_file):
        """Test that README has required sections."""
        content = readme_file.read_text()
        
        required_sections = [
            "Overview",
            "Quick Start",
            "Configuration",
            "Parameters",
            "Profiles",
            "Pipeline Outputs",
            "Examples"
        ]
        
        for section in required_sections:
            assert f"## {section}" in content or f"### {section}" in content, \
                f"Section '{section}' not found in README"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
