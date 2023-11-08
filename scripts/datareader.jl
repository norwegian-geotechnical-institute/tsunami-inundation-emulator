module DataReader

using Distributed: myid, RemoteChannel
using NCDatasets
using Flux: MaxPool
using YAML: load_file
using CUDA: CuArray, cu

export Reader, parse_config

struct Reader
    data_dir::String                                    # Path to root directory of the data.
    # scenarios_file::String                              # File with list of scenarios. 
    grid_file::String                                   # Topography file.  
    batch_size::Int                                     # Size of batches.
    ct_slice::Tuple{UnitRange{Int}, UnitRange{Int}}     # Slice used to subset the inundation raster.
    ts_slice::Tuple{UnitRange{Int}, UnitRange{Int}}     # Slice used to subset timeseries
    scale::Tuple{Int,Int}                               # Downsampling scale.  Maximum value over windiw with size = (scale[1], scale[2]).
    dims::Tuple{Int,Int}                                # Dimension of the final subsampled 
    topography::Array{Float32, 2}                       # Topography     
    gpu::Bool                                           # Return CuArray or regular Array.

    function Reader(config::Dict)
        ct_slice = config["ct_slice"]
        if haskey(config, "scale")
            scale = config["scale"]
        else
            scale = (1,1)
        end
        dims = (length(ct_slice[1]) ÷ scale[1],length(ct_slice[2]) ÷ scale[2])
        
        if haskey(config, "gpu")
            gpu = config["gpu"]
        else
            gpu = false
        end
        # Load topography
        topography = NCDataset(config["grid_file"], "r")["z"][ct_slice...]
        
        # Create object.
        new(config["data_dir"],
            config["grid_file"],
            config["batch_size"],
            ct_slice,
            config["ts_slice"],
            scale,
            dims,
            topography,
            gpu)
    end

    function Reader(configfile::String)
        # Read config file and set constants.
        config = parse_config(configfile)
        Reader(config)
    end
end

function parse_config(configfile::String)
    config = load_file(configfile)
    
    function parse_tuple_range(tuple_range_string)
        ct_list = [parse(Int, x.match) for x in eachmatch(r"[0-9]+", tuple_range_string; overlap=false)]
        return ct_list[1]:ct_list[2], ct_list[3]:ct_list[4]
    end

    function parse_tuple(tuple_string)
        return Tuple([parse(Int, x.match) for x in eachmatch(r"[0-9]+", tuple_string; overlap=false)])
    end

    # Some types are correctly loaded by YAML! Need to parse..
    for key in ["ct_slice", "ts_slice"] 
        if haskey(config, key)
            config[key] = parse_tuple_range(config[key])
        end
    end
    for key in ["scale", "dims"]
        if haskey(config, key)
            config[key] = parse_tuple(config[key])
        end
    end
    
    return(config)
end

function scenarios(scenarios_file)
    return RemoteChannel(()->Channel{String}(ch -> load_scenarios(ch, scenarios_file)))
end

function load_scenarios(channel, scenarios_file)
    epoch = 1
    while true
        @info "Reads epoch: $epoch"
        for scenario in eachline(scenarios_file)
            put!(channel, scenario)
        end
        epoch += 1
    end
end

function (reader::Reader)(scenarios::RemoteChannel{Channel{String}}, batches::RemoteChannel{Channel{Any}})
    @info("load_batches")
    if reader.gpu
        flow_depths = CuArray{Float32}(undef, reader.dims..., 1, reader.batch_size)
        deformed_topographies = CuArray{Float32}(undef, reader.dims..., 1, reader.batch_size)
        etas = CuArray{Float32}(undef, length(reader.ts_slice[1]), length(reader.ts_slice[2]), 1, reader.batch_size)
    else
        flow_depths = Array{Float32}(undef, reader.dims..., 1, reader.batch_size)
        deformed_topographies = Array{Float32}(undef, reader.dims..., 1, reader.batch_size)
        etas = Array{Float32}(undef, length(reader.ts_slice[1]), length(reader.ts_slice[2]), 1, reader.batch_size)
    end 
    scenario_names = Array{String, 1}(undef, reader.batch_size)
    index = 1
    worker_id = myid()
    # Create batches/batches.
    while true
        scenario = take!(scenarios)
        flow_depth, eta, deformed_topography = get_sample(reader, scenario)
        if reader.gpu
            flow_depths[:,:,1,index] = cu(downsample(reader, flow_depth))
            deformed_topographies[:,:,1,index] = cu(downsample(reader, deformed_topography))
            etas[:,:,1,index] = cu(eta)
        else
            flow_depths[:,:,1,index] = downsample(reader, flow_depth)
            deformed_topographies[:,:,1,index] = downsample(reader, deformed_topography)
            etas[:,:,1,index] = eta
        end
        scenario_names[index] = scenario
        if index % reader.batch_size == 0
            @debug "Delivering batch."
            put!(batches, (; flow_depths, etas, deformed_topographies, worker_id, scenario_names))
            index = 0
        end
        index = index + 1
    end
    # Last batch.
    etas = eta[:,:,:,1:index-1]
    flow_depths = flow_depths[:,:,:,1:index-1]
    deformed_topographies = deformed_topographies[:,:,:,1:index-1]
    scenario_names[index] = scenario
    put!(batches, (; flow_depths, etas, deformed_topographies, worker_id, scenario_names))
end

function (reader::Reader)(scenario::String)
    flow_depth, deformed_topography = get_flowdepth(reader, joinpath(reader.data_dir, scenario*"_CT_10m.nc"))
    flow_depth = downsample(reader, flow_depth)
    deformed_topography = downsample(reader, deformed_topography)
    eta = get_timeseries(reader, joinpath(reader.data_dir, scenario*"_ts.nc"))
    return(flow_depth, eta, deformed_topography)
end

# Not in use (replaced by get_sample)
function get_timeseries(reader, filename_ts)
    eta = Array{Float32}(undef, length(reader.ts_slice[1]), length(reader.ts_slice[2])) # For CUDA to infer type. Better way??
    eta[:,:] = NCDataset(filename_ts, "r")["eta"][reader.ts_slice...]
    return eta
end


function get_sample(reader, scenario)
    filename_CT = joinpath(reader.data_dir, scenario*"_CT_10m.nc")
    filename_ts = joinpath(reader.data_dir, scenario*"_ts.nc")

    eta = Array{Float32}(undef, length(reader.ts_slice[1]), length(reader.ts_slice[2])) # For CUDA to infer type. Better way??
    eta[:,:] = NCDataset(filename_ts, "r")["eta"][reader.ts_slice...]

    flow_depth =  zeros(Float32, (length(reader.ct_slice[1]), length(reader.ct_slice[2])))
    deformed_topography =  Array{Float32}(undef, length(reader.ct_slice[1]), length(reader.ct_slice[2]))
    NCDataset(filename_CT, "r") do ds
        max_height =  ds["max_height"][reader.ct_slice...]
        deformation = ds["deformation"][reader.ct_slice...]
        deformed_topography[:,:] = reader.topography - deformation

        mask = reader.topography .> 0 .&& max_height .!== missing .&& max_height .> deformed_topography
        flow_depth[mask] = (max_height .- deformed_topography)[mask]
    end
    return flow_depth, eta, deformed_topography
end

function get_flowdepth(reader, filename_CT)
    flow_depth =  zeros(Float32, (length(reader.ct_slice[1]), length(reader.ct_slice[2])))
    deformed_topography =  Array{Float32}(undef, length(reader.ct_slice[1]), length(reader.ct_slice[2]))
    NCDataset(filename_CT, "r") do ds
        max_height =  ds["max_height"][reader.ct_slice...]
        deformation = ds["deformation"][reader.ct_slice...]
        deformed_topography[:,:] = reader.topography - deformation

        mask = reader.topography .> 0 .&& max_height .!== missing .&& max_height .> deformed_topography
        flow_depth[mask] = (max_height .- deformed_topography)[mask]
    end
    return flow_depth, deformed_topography
end

function downsample(reader, x)
    x = Array{Float32}(reshape(x, (size(reader.topography)...,1,1)))
    return MaxPool(reader.scale)(x)[:,:,1,1]
end

end # DataReader