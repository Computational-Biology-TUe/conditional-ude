# Perform train test split, and prepare data into a convenient JLD2 file
using CSV, DataFrames, JLD2, StableRNGs, StatsBase, HypothesisTests
rng = StableRNG(270523)

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

#### SECTION 1: Ohashi data ####

# read the ohashi data
data = DataFrame(CSV.File("data/ohashi_csv/ohashi_OGTT.csv"))
data_filtered = dropmissing(data)

subject_info = DataFrame(CSV.File("data/ohashi_csv/ohashi_subjectinfo.csv"))

# create the time series
subject_numbers = data_filtered[!,:No]
subject_info_filtered = subject_info[subject_info[!,:No] .∈ Ref(subject_numbers), :]
types = String.(subject_info_filtered[!,:type])
timepoints = [0.0, 30.0, 60.0, 90.0, 120.0]
glucose_indices = 2:6
cpeptide_indices = 12:16
ages = subject_info_filtered[!,:age]
body_weights = subject_info_filtered[!,:BW]
bmis = subject_info_filtered[!,:BMI]

glucose_data = Matrix{Float64}(data_filtered[:, glucose_indices]) .* 0.0551 # convert to mmol/L
cpeptide_data = Matrix{Float64}(data_filtered[:, cpeptide_indices]) .* 0.3311 # convert to nmol/L

# test age differences
ngt_igt = pvalue(MannWhitneyUTest(ages[types .== "NGT"], ages[types .== "IGT"]))
t2dm_igt = pvalue(MannWhitneyUTest(ages[types .== "T2DM"], ages[types .== "IGT"]))
t2dm_ngt = pvalue(MannWhitneyUTest(ages[types .== "T2DM"], ages[types .== "NGT"]))


# figure illustrating the OGTT data
figure_ogtt = let f = Figure(size=(550,300))

    ga = GridLayout(f[1,1])
    gb = GridLayout(f[1,2])

    ax_glucose = Axis(ga[1,1], xlabel="Time (min)", ylabel="Glucose (mM)")
    ax_cpeptide = Axis(gb[1,1], xlabel="Time (min)", ylabel="C-peptide (nM)")
    markers=['●', '▴', '■']
    markersizes = [10, 18, 10]
    for ((i,type), marker, markersize) in zip(enumerate(unique(types)), markers, markersizes)
        type_indices = types .== type
        mean_glucose = mean(glucose_data[type_indices,:], dims=1)[:]
        std_glucose = 1.96 .* std(glucose_data[type_indices,:], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_glucose, timepoints, mean_glucose .- std_glucose, mean_glucose .+ std_glucose, color=(COLORS[type], 0.3), label=type)
        lines!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)

        mean_cpeptide = mean(cpeptide_data[type_indices,:], dims=1)[:]
        std_cpeptide = 1.96 .* std(cpeptide_data[type_indices,:], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_cpeptide, timepoints, mean_cpeptide .- std_cpeptide, mean_cpeptide .+ std_cpeptide, color=(COLORS[type], 0.3), label=type)
        lines!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)
    end
    Legend(f[2,1:2], ax_glucose, orientation=:horizontal, merge=true)


    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end

    f
end

save("figures/illustration_ogtt.eps", figure_ogtt, px_per_unit=4)

clamp_indices = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_indices.csv"))

clamp_indices_filtered = clamp_indices[clamp_indices[!,:No] .∈ Ref(subject_numbers), :]
disposition_indices = clamp_indices_filtered[!, Symbol("clamp PAI")]
first_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10)")]
second_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10-90)")]
isi = clamp_indices_filtered[!, Symbol("ISI(GIR/Glu/IRI)")]
total = first_phase .+ second_phase 

f_train = 0.70

training_indices, testing_indices = let types = types
    training_indices = Int[]
    for type in unique(types)
        type_indices = findall(types .== type)
        n_train = Int(round(f_train * length(type_indices)))
        selection = StatsBase.sample(rng, type_indices, n_train, replace=false)
        append!(training_indices, selection)
    end
    training_indices = sort(training_indices)
    testing_indices = setdiff(1:length(types), training_indices)
    training_indices, testing_indices
end

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi.jld2";
    train = (
        glucose=glucose_data[training_indices,:], 
        cpeptide=cpeptide_data[training_indices,:], 
        subject_numbers=subject_numbers[training_indices], 
        types=types[training_indices], 
        timepoints=timepoints, 
        ages=ages[training_indices], 
        body_weights=body_weights[training_indices],
        bmis=bmis[training_indices],
        disposition_indices=disposition_indices[training_indices], 
        first_phase=first_phase[training_indices],
        second_phase=second_phase[training_indices],
        total_insulin=total[training_indices],
        insulin_sensitivity=isi[training_indices]
    ),
    test = (
        glucose=glucose_data[testing_indices,:], 
        cpeptide=cpeptide_data[testing_indices,:], 
        subject_numbers=subject_numbers[testing_indices], 
        types=types[testing_indices], 
        timepoints=timepoints, 
        ages=ages[testing_indices], 
        body_weights=body_weights[testing_indices],
        bmis=bmis[testing_indices],
        disposition_indices=disposition_indices[testing_indices], 
        first_phase=first_phase[testing_indices],
        second_phase=second_phase[testing_indices],
        total_insulin=total[testing_indices],
        insulin_sensitivity=isi[testing_indices]
    )
)

# illustration of clamp data
clamp_data = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_blood.csv", delim=';', decimal=','))

clamp_data_filtered = clamp_data[clamp_data[!,:No] .∈ Ref(subject_numbers), :]
clamp_insulin_data = Matrix{Float64}(clamp_data_filtered[:, 12:18])
clamp_insulin_timepoints = [0,5,10,15,60,75,90]

figure_clamp_insulin = let fig
    fig = Figure(size=(400,400))
    ax = Axis(fig[1,1], xlabel="Time (min)", ylabel="Insulin (mU/L)")
    for (i, type) in enumerate(["NGT", "T2DM"])
        type_indices = types .== type
        mean_insulin = mean(clamp_insulin_data[type_indices,:], dims=1)[:]
        std_insulin = std(clamp_insulin_data[type_indices,:], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax, clamp_insulin_timepoints, repeat([mean_insulin[1]], length(mean_insulin)), mean_insulin, color=(Makie.ColorSchemes.tab10[i], 0.3), label=type)
        lines!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
        scatter!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
    end

    vlines!(ax, [10], color=:black, linestyle=:dash, linewidth=1)
    text!(ax, -12, 60;text="1ˢᵗ phase")
    text!(ax, 45, 60;text="2ⁿᵈ phase")



    Legend(fig[2,1], ax, orientation=:horizontal, merge=true)
    fig
end

save("figures/supplementary/illustration_clamp_insulin.png", figure_clamp_insulin, px_per_unit=4)


#### SECTION 2: Fujita data ####

# read the fujita data
data = DataFrame(CSV.File("data/fujita_csv/fujita_ogtt.csv"))

timepoints = parse.(Int64, names(data)[3:end-1])
glucose_data = Matrix{Float64}(data[data[!,:Molecule] .== "Glucose", 3:end-1]) .* 0.0551 # convert to mmol/L
cpeptide_data = Matrix{Float64}(data[data[!,:Molecule] .== "C-peptide", 3:end-1]).* 0.3311 # convert to nmol/L
ages = repeat([29], size(glucose_data, 1))

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/fujita.jld2";
    glucose=glucose_data, 
    cpeptide=cpeptide_data, 
    timepoints=timepoints, 
    ages=ages
)

