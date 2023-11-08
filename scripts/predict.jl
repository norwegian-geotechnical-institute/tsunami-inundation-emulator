import Pkg; Pkg.instantiate()
using NCDatasets, Flux, CUDA, Distributed, DataFrames, CSV, Serialization
using Statistics, Random, StatsBase
import YAML
using DelimitedFiles: readdlm, writedlm
using ArgParse
using DataStructures

"""
Script to make predictions. The main input is the model specified by the path to the model folder. Next is a text file with one scenario
per line, specified by the relative path of the scenario files subject to the models datadir (specified in the models config.yml).

julia --project predict.jl /path/to/folder /path/to/scenario file --output-dir /path/to/output
"""

s = ArgParseSettings()

@add_arg_table s begin
	"model-path"
	    help = "Path to the model folder."
	    arg_type = String
	"scenarios"
	    help = "Path to text file listing all scenarios."
	    arg_type = String
	"--output-dir"
	    help = "Store output in directory."
	    arg_type = String
	    default = pwd()
end

args = parse_args(s)

# Retrieve the values
model_path = args["model-path"]
scenarios_file = args["scenarios"]
output_dir = args["output-dir"]
call_directory = pwd()

# Move to model directory.
cd(model_path)

# If scenarios file, and output-dir was given as relative paths.
if !isfile(scenarios_file)
	scenarios_file = joinpath(call_directory, scenarios_file)
end
if !isdir(output_dir)
	output_dir = joinpath(call_directory, output_dir)
end

@info "model-path: $model_path, scenarios: $scenarios_file, output directory: $output_dir, call_directory: $call_directory"

# Load files from model directory.
@info "Includes model_config.jl and datareader.jl from directory: $model_path."
include( joinpath(model_path,"model_config.jl") )
include( joinpath(model_path,"datareader.jl") )

config = DataReader.parse_config("config.yml")
reader = DataReader.Reader(config)
model = ModelConfig.load("$(config["model_name"]).jls")
ct_mask = BitArray(readdlm("ct_mask.txt", '\t',Bool, '\n'))

function main()
	for (index, scenario) in enumerate(eachline(scenarios_file))
		@info "Make prediction for scenario: $scenario"
        x, hat_x = get_prediction(scenario)
		attributes = OrderedDict("scenario" => scenario)
		
		Dataset(joinpath(output_dir,"prediction_$index.nc"),"c", attrib = attributes) do ds
			defDim(ds,"x",size(hat_x)[1])
			defDim(ds,"y",size(hat_x)[2])
			preds = defVar(ds,"predicted-flow-depth", Float32,("x","y"))
			preds[:,:] = hat_x
			residuals = defVar(ds, "residual", Float32, ("x", "y"))
			residuals[:,:] = x - hat_x
		end
    end
end

function get_prediction(scenario)
    x, eta, deformed_topography = reader(scenario) # Reads scenario.

    # Make prediction.
    eta = reshape(eta, (length(config["ts_slice"][1]), length(config["ts_slice"][2]),1,1))
    hat_x = zeros(Float32, config["dims"]...);
    
    Flux.reset!(model)
    hat_x[ct_mask] = relu(model(eta) - deformed_topography[ct_mask]);

    return x, hat_x
end

# Call the main function
main()
