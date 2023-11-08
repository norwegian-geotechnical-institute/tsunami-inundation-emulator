using NCDatasets, Flux, CUDA, Distributed, DataFrames, CSV
using Statistics: mean
using DelimitedFiles: readdlm

addprocs(3; exeflags="--project")

@everywhere begin
    using Logging
    global_logger(SimpleLogger(stdout, Logging.Info))
    include("datareader.jl")
    include("model_config.jl")
    config = DataReader.parse_config("config.yml")

    # Set here
    config["gpu"] = false
    config["batch_size"] = 5
end

# Load mask
const ct_mask = BitArray(readdlm("ct_mask.txt", '\t',Bool, '\n'))

function main()
    if ARGS[1] == "train"
        dataset = config["train_data"]
        eval_dir = joinpath(config["rundir"], "evaluation", "train")
        calculate_r_squared(dataset, eval_dir)
    elseif ARGS[1] == "test"
        dataset = config["test_data"]
        eval_dir = joinpath(config["rundir"], "evaluation", "test")
        calculate_r_squared(dataset, eval_dir)
    else
        println("Select either train or test.")
    end
end

function calculate_r_squared(dataset, eval_dir)
    @info "Calculates R-squared: $dataset."
    # Load model
    model = ModelConfig.load(joinpath(config["rundir"],"$(config["model_name"]).jls"))
    Flux.reset!(model)

    # Create evaluation directory
    if !isdir(eval_dir)
        mkpath(eval_dir)
    end
    
    # Load batches on workers.
    @everywhere begin
    reader = DataReader.Reader(config)
    end
    batches = RemoteChannel(()->Channel(4))
    scenarios = DataReader.scenarios(dataset)
    
    for worker in workers()
        remote_do(reader, worker, scenarios, batches)
    end

    nr_of_batches = ceil(countlines(dataset)/config["batch_size"])
    
    @info "First pass. Computing mean and count hits."
    mean_target = zeros(Float32, sum(ct_mask)) # Mean of target value
    count_hits = zeros(Int32, sum(ct_mask))    # Nr of times the target value has nonzero flow depth.

    for batch_nr in 1:nr_of_batches
        batch = take!(batches)
        #@info "Batch: $batch_nr"
        
        hit = sum(batch.flow_depths[ct_mask,1,:] .> 0., dims=2)
        target = batch.flow_depths[ct_mask,1,:]
        preds = relu(model(batch.etas) - batch.deformed_topographies[ct_mask,1,:])
    
        mean_target = mean(target, dims=2)/batch_nr .+ mean_target*((batch_nr - 1)/batch_nr)
        count_hits = count_hits .+ hit
    end

    @info "Second pass. Computing (mean) total squares and (mean) residual squares."
    mean_residual_squares = zeros(Float32, sum(ct_mask))
    mean_total_squares = zeros(Float32, sum(ct_mask))

    for batch_nr in 1:nr_of_batches
        batch = take!(batches)

        #@info "Batch: $batch_nr"
        target = batch.flow_depths[ct_mask,1,:]
        preds = relu(model(batch.etas) - batch.deformed_topographies[ct_mask,1,:])

        residual_square = (target-preds).^2
        total_square = (target-mean_target*ones(config["batch_size"])').^2

        mean_residual_squares = mean(residual_square, dims=2)/batch_nr .+ mean_residual_squares*((batch_nr - 1)/batch_nr)
        mean_total_squares = mean(total_square, dims=2)/batch_nr .+ mean_total_squares*((batch_nr - 1)/batch_nr)
    end
    
    @info "Calculates R-squared."
    rsquared = 1 .- mean_residual_squares./mean_total_squares

    @info "Writes to file: $(joinpath(eval_dir, "r_squared.nc"))"
    
    rsquared_map = zeros(Float32, config["dims"])
    fill!(rsquared_map, NaN)
    rsquared_map[ct_mask] = rsquared

    hit_map = zeros(Float32, config["dims"]);
    hit_map[ct_mask] = count_hits
    
    Dataset(joinpath(eval_dir,"r_squared.nc"),"c") do ds
        defDim(ds,"x",config["dims"][1])
        defDim(ds,"y",config["dims"][2])
        r = defVar(ds,"r-squared", Float32,("x","y"))
        r[:,:] = rsquared_map
        c = defVar(ds,"hit-count", Int32,("x","y"))
        c[:,:] = hit_map
    end
end

@time main()
