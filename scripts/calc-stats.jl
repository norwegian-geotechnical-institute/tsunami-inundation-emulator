using NCDatasets, DataFrames, CSV, JSON
using Statistics, StatsBase

function compute_stats()
    # Load summary results into dataframe.
    eval_df = CSV.File("summary_results.txt"; delim='\t') |> DataFrame;
    unique!(eval_df, :scenario)

    # Dictionaryt for collection of statistics
    stats = Dict{String, Float64}()
    
    stats["mean_l2"] = mean(eval_df.l2_err)
    stats["mean_l1"] = mean(eval_df.l1_err)
    stats["max_l2"] = maximum(eval_df.l2_err)
    stats["max_l1"] = maximum(eval_df.l1_err)
    stats["q95_l2"] = quantile(eval_df.l2_err, 0.95)
    stats["q95_l1"] = quantile(eval_df.l1_err, 0.95)

    # Load Aida and residual statistics.
    aida_df = CSV.File("summary_by_class.txt"; delim='\t', missingstring="missing") |> DataFrame

    for q = [5, 50, 95]
        stats["aida_K_q$(q)"] = quantile(aida_df[aida_df.nr_of_samples_aida .> 100 .&& aida_df.class .> 1,:].K, q/100)
        stats["aida_kappa_q$(q)"] = quantile(aida_df[aida_df.nr_of_samples_aida .> 100 .&& aida_df.class .> 1,:].kappa, q/100)
        stats["mean_res_q$(q)"] = quantile(aida_df[aida_df.nr_of_samples_res .> 100,:].mean_res, q/100)
        stats["std_res_q$(q)"] = quantile(aida_df[aida_df.nr_of_samples_res .> 100,:].std_res, q/100)
    end
    return stats
end

function main()
    println("Calculates stats for $(pwd())")
    model_stats = Dict()
    model_stats["eval_dir"] = pwd()
    model_stats["stats"] = compute_stats()
    open("model_stats.json","w") do f
        JSON.print(f, model_stats)
    end
end

main()
