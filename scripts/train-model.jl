using Distributed: addprocs, workers, rmprocs, myid, RemoteChannel, remote_do, @everywhere
#using CSV: write, File
#using DataFrames: DataFrame
#using Flux: mse, params, flatten, ADAM, gradient, Conv, Chain, Dense, MaxPool, relu, leakyrelu, gpu, ConvTranspose, update!, cpu
using NNlib, Flux, CUDA, Random
# Need to add like this due to BSON.@load
using Dates: format, now
using BSON: @load, @save
addprocs(6; exeflags="--project")

@everywhere begin
using Logging
include("datareader.jl")
include("model_config.jl")

global_logger(SimpleLogger(stdout, Logging.Info))
end

using DelimitedFiles: readdlm, writedlm
import YAML

# Read config file and set constants.
const config = YAML.load_file("config.yml")
const train_data = config["train_data"]
const test_data = config["test_data"]
const rundir = config["rundir"]
const model_name = config["model_name"]
const max_batch_nr = config["max_batch_nr"]
const reg = config["reg"]

# Load mask
const ct_mask = BitArray(readdlm("ct_mask.txt", '\t',Bool, '\n'))

function main()

    model = ModelConfig.load(joinpath(rundir,"$(model_name).jls"))
    @load "optimizer.bson" opt
    
    if config["gpu"]
        model = gpu(model)
    end
    
    @info model

    # Loss
    l2(x) = Flux.mean(x.^2)
    l1(x) = Flux.mean(abs.(x))
    
    function loss(flow_depth, eta, deformed_topography)
        Flux.reset!(model)
        # Switch loss by applying leakyrelu (rel) or not.
        y_hat = Flux.leakyrelu(model(eta) - deformed_topography[ct_mask,1,:])
        y = flow_depth[ct_mask,1,:]
        l2(Flux.flatten(y_hat) .- Flux.flatten(y)) + reg*sum(l2, Flux.params(model))
        # Regularize encoder more heavily.
        #l2(Flux.flatten(y_hat) .- Flux.flatten(y)) + 1e-1*sum(l2, Flux.params(model[1])) + 1e-2*sum(l2, Flux.params(model[2]))
    end
    
    ps = Flux.params(model) # model parameters for optimization

    # Train model
    write_header = !isfile("train-summary.txt") 

    open("train-summary.txt", "a") do io_summary
        if write_header
            writedlm(io_summary, ["epoch" "batch" "processed_scenarios" "train_loss" "test_loss"])
        end
        train(model, loss, opt, ps, io_summary)
    end
    rmprocs(workers()) # Delete worker processes.
    
    ModelConfig.save(model, joinpath(rundir,"$(model_name).jls"))
    @info "Writes optimizer to file: optimizer.bson"
    @save joinpath(rundir, "optimizer.bson") opt

    @info "Train summary written to file: train-summary.csv"
end

function train(model, loss, opt, ps, io_summary)

    @everywhere begin
        #reader_config = get_readerconfig()
        reader = DataReader.Reader("config.yml")
    end

    @info "Start loading batches with scenarios."
    train_scenarios = DataReader.scenarios(train_data)
    test_scenarios = DataReader.scenarios(test_data)
    train_batches = RemoteChannel(()->Channel(2))
    test_batches = RemoteChannel(()->Channel(2))
    for worker in workers()[1:end-1]
        remote_do(reader, worker, train_scenarios, train_batches)
    end
    remote_do(reader, workers()[end], test_scenarios, test_batches)

    nr_of_scenarios = countlines(train_data) 
    processed_scenarios = 0
    batch_nr = 0
    while batch_nr < max_batch_nr   
        batch = take!(train_batches)
        # Update counters
        batch_nr += 1
        processed_scenarios += size(batch.flow_depths)[4]
        epoch = ceil(Int32, processed_scenarios/nr_of_scenarios)
        
        # Train on batch
        grads = gradient(ps) do
            loss(batch.flow_depths, batch.etas, batch.deformed_topographies)
        end
        # Update the parameters based on the chosen
        # optimiser (opt)
        Flux.update!(opt, ps, grads)

        # Write values
        if (batch_nr-1) % 10 == 0
            test_batch = take!(test_batches)
            test_loss = loss(test_batch.flow_depths, test_batch.etas, batch.deformed_topographies)
            train_loss = loss(batch.flow_depths, batch.etas, batch.deformed_topographies)
            @info "Epoch: $epoch, processed_scenarios: $processed_scenarios, Train Loss: $train_loss, Test Loss: $test_loss"
            writedlm(io_summary, [epoch batch_nr processed_scenarios train_loss test_loss])
        end
        if (batch_nr-1) % 100 == 0
            # Write summary to file.
            flush(io_summary)
        end
        if batch_nr % 1e4 == 0
          # Write model to file
          ModelConfig.save(model, joinpath(rundir,"$(model_name)_$(batch_nr).jls"))
        end
    end
end

@time main()
