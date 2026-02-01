Implemented an end-to-end Nextflow pipeline for the discovery process, including data preparation, enrichment, surrogate training, generative sampling, DFT validation, analysis, and synthesis design.

#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
 * End-to-End Discovery Pipeline
 * Quantum Materials Discovery Platform
 */

// Define parameters
params.raw_dataset = "data/qm9_raw.pt"
params.output_dir = "results/pipeline_output"
params.num_samples = 100
params.num_epochs_surrogate = 50
params.num_epochs_score = 100

// Process 1: Data Preparation
process data_preparation {
    publishDir "${params.output_dir}/data", mode: 'copy'
    
    input:
    path raw_dataset
    
    output:
    path 'processed_dataset.pt', emit: processed_data
    
    script:
    """
    python ${projectDir}/projects/phononic-discovery/framework/scripts/01_prepare_data.py \
        --input ${raw_dataset} \
        --output processed_dataset.pt
    """
}

// Process 2: xTB Enrichment
process xtb_enrichment {
    publishDir "${params.output_dir}/enriched", mode: 'copy'
    
    input:
    path processed_dataset
    
    output:
    path 'enriched_dataset.pt', emit: enriched_data
    
    script:
    """
    python ${projectDir}/projects/phononic-discovery/framework/scripts/02_enrich_dataset.py \
        --input-path ${processed_dataset} \
        --output-path enriched_dataset.pt
    """
}

// Process 3: Surrogate Training
process surrogate_training {
    publishDir "${params.output_dir}/models/surrogate", mode: 'copy'
    
    input:
    path enriched_dataset
    
    output:
    path 'surrogate_model.pt', emit: surrogate_model
    path 'training_metrics.json', emit: surrogate_metrics
    
    script:
    """
    python ${projectDir}/projects/phononic-discovery/framework/scripts/03_train_surrogate.py \
        --dataset-path ${enriched_dataset} \
        --num-epochs ${params.num_epochs_surrogate} \
        --output-dir . \
        --save-model surrogate_model.pt
    """
}

// Process 4: Score Model Training
process score_training {
    publishDir "${params.output_dir}/models/score", mode: 'copy'
    
    input:
    path enriched_dataset
    
    output:
    path 'score_model.pt', emit: score_model
    path 'training_metrics.json', emit: score_metrics
    
    script:
    """
    python ${projectDir}/projects/phononic-discovery/framework/scripts/04_train_score_model_v2.py \
        --dataset-path ${enriched_dataset} \
        --num-epochs ${params.num_epochs_score} \
        --output-dir . \
        --save-model score_model.pt
    """
}

// Process 5: Generative Sampling
process generative_sampling {
    publishDir "${params.output_dir}/generated", mode: 'copy'
    
    input:
    path score_model
    path surrogate_model
    
    output:
    path 'generated_molecules/*.xyz', emit: generated_structures
    path 'generation_log.txt', emit: generation_log
    
    script:
    """
    mkdir -p generated_molecules
    python ${projectDir}/projects/phononic-discovery/framework/scripts/06_generate_molecules.py \
        --score-model ${score_model} \
        --surrogate-model ${surrogate_model} \
        --num-samples ${params.num_samples} \
        --output-dir generated_molecules \
        > generation_log.txt
    """
}

// Process 6: DFT Validation
process dft_validation {
    publishDir "${params.output_dir}/validated", mode: 'copy'
    
    input:
    path xyz_files
    
    output:
    path 'validation_results/*.json', emit: validation_results
    path 'validated_structures/*.xyz', emit: validated_structures
    
    script:
    """
    mkdir -p validation_results validated_structures
    python ${projectDir}/projects/phononic-discovery/framework/dft_validation/run_gpaw_validation.py \
        --input-dir ${xyz_files.parent} \
        --output-dir validation_results \
        --validated-output validated_structures
    """
}

// Process 7: Analysis and Visualization
process analysis_visualization {
    publishDir "${params.output_dir}/analysis", mode: 'copy'
    
    input:
    path validation_results
    path enriched_dataset
    
    output:
    path 'figures/*.png', emit: figures
    path 'analysis_report.html', emit: report
    
    script:
    """
    mkdir -p figures
    python ${projectDir}/projects/phononic-discovery/framework/scripts/analyze_enriched_dataset.py \
        --dataset ${enriched_dataset} \
        --validation ${validation_results} \
        --output-dir figures \
        --report analysis_report.html
    """
}

// Process 8: Advanced Benchmarking
process advanced_benchmarking {
    publishDir "${params.output_dir}/benchmark", mode: 'copy'
    
    input:
    path enriched_dataset
    path surrogate_model
    path score_model
    
    output:
    path 'benchmark_results.json', emit: benchmark_results
    path 'benchmark_plots/*.png', emit: benchmark_plots
    
    script:
    """
    mkdir -p benchmark_plots
    python ${projectDir}/projects/phononic-discovery/framework/scripts/05_advanced_benchmark.py \
        --dataset ${enriched_dataset} \
        --surrogate ${surrogate_model} \
        --score ${score_model} \
        --output benchmark_results.json \
        --plots-dir benchmark_plots
    """
}

// Workflow definition
workflow {
    // Input channel
    raw_data = Channel.fromPath(params.raw_dataset)
    
    // Execute pipeline
    data_preparation(raw_data)
    xtb_enrichment(data_preparation.out.processed_data)
    
    // Parallel training
    surrogate_training(xtb_enrichment.out.enriched_data)
    score_training(xtb_enrichment.out.enriched_data)
    
    // Generation with trained models
    generative_sampling(
        score_training.out.score_model,
        surrogate_training.out.surrogate_model
    )
    
    // Validation (optional - can be commented out for faster testing)
    // dft_validation(generative_sampling.out.generated_structures)
    
    // Analysis
    analysis_visualization(
        generative_sampling.out.generation_log,
        xtb_enrichment.out.enriched_data
    )
    
    // Benchmarking
    advanced_benchmarking(
        xtb_enrichment.out.enriched_data,
        surrogate_training.out.surrogate_model,
        score_training.out.score_model
    )
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: $workflow.success"
    println "Duration: $workflow.duration"
}