import PyQDecoders

using CircuitCompilation2xn
using CircuitCompilation2xn: add_two_qubit_gate_noise, evaluate_code_decoder_shor_syndrome, evaluate_code_decoder_naive_syndrome
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5, CSS, PyBeliefPropOSDecoder
using CairoMakie

using QuantumClifford.ECC: AbstractSyndromeDecoder, TableDecoder, evaluate_decoder, Toric, PyMatchingDecoder

function my_plot_both_synd(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_shift=0.0001, p_wait=1-exp(-14.5/28_000); name=string(typeof(code)))
    title = name*" Code - Naive Syndrome Circuit"
    checks = parity_checks(code)
    scirc, _ = naive_syndrome_circuit(checks)
    #ecirc = naive_encoding_circuit(code)
    ecirc = nothing
    nsamples = 20_000
    gate_fidelity = 0.9995
    m = 10 # improvment factor
    #gate_noise = (1 - gate_fidelity)/m #improvement in gate fidelity
    gate_noise = 1- gate_fidelity
    #p_wait = 1-exp(-14.5/m/28_000) # improvement in time to shuttle
    #p_shift = p_shift/m #improvement in shuttling fidelity

    #gate_noise = 0
    # p_wait = 0
    #p_shift = 0
    error_rates = exp10.(range(-4,-0.5,length=30))


    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA = [post_ec_error_rates_MA_CA[i][1] for i in eachindex(post_ec_error_rates_MA_CA)]
    z_error_MA_CA = [post_ec_error_rates_MA_CA[i][2] for i in eachindex(post_ec_error_rates_MA_CA)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB= [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB = [post_ec_error_rates_MA_CB[i][1] for i in eachindex(post_ec_error_rates_MA_CB)]
    z_error_MA_CB = [post_ec_error_rates_MA_CB[i][2] for i in eachindex(post_ec_error_rates_MA_CB)]

    # Naive compilation and shuttle noise -> no gate noise and gate noise == init noise
    post_ec_error_rates_MB_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MB_CA = [post_ec_error_rates_MB_CA[i][1] for i in eachindex(post_ec_error_rates_MB_CA)]
    z_error_MB_CA = [post_ec_error_rates_MB_CA[i][2] for i in eachindex(post_ec_error_rates_MB_CA)]

    # Naive compilation and shuttle noise - gate noise == init noise 
    post_ec_error_rates_MB_CB = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MB_CB = [post_ec_error_rates_MB_CB[i][1] for i in eachindex(post_ec_error_rates_MB_CB)]
    z_error_MB_CB = [post_ec_error_rates_MB_CB[i][2] for i in eachindex(post_ec_error_rates_MB_CB)]

    # Gate shuffled circuit 
    gate_shuffle_circ = CircuitCompilation2xn.gate_Shuffle!(scirc)

    # Gate shuffle - no gate noise
    post_ec_error_rates_MS_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, gate_shuffle_circ, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MS_CA = [post_ec_error_rates_MS_CA[i][1] for i in eachindex(post_ec_error_rates_MS_CA)]
    z_error_MS_CA = [post_ec_error_rates_MS_CA[i][2] for i in eachindex(post_ec_error_rates_MS_CA)]

    # Gate shuffle- gate noise== init noise
    post_ec_error_rates_MS_CB = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, gate_shuffle_circ, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MS_CB = [post_ec_error_rates_MS_CB[i][1] for i in eachindex(post_ec_error_rates_MS_CB)]
    z_error_MS_CB = [post_ec_error_rates_MS_CB[i][2] for i in eachindex(post_ec_error_rates_MS_CB)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA = [post_ec_error_rates_MC_CA[i][1] for i in eachindex(post_ec_error_rates_MC_CA)]
    z_error_MC_CA = [post_ec_error_rates_MC_CA[i][2] for i in eachindex(post_ec_error_rates_MC_CA)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MC_CB = [post_ec_error_rates_MC_CB[i][1] for i in eachindex(post_ec_error_rates_MC_CB)]
    z_error_MC_CB = [post_ec_error_rates_MC_CB[i][2] for i in eachindex(post_ec_error_rates_MC_CB)]

    f = Figure(size=(1500, 1500))
    # X plot
    f_x =  f[1,1]
    ax = f[1,1] = Axis(f_x, xlabel="p_mem = physical qubit error rate after encoding",ylabel="Logical error rate",title=title*" Logical X")
    lines!(f_x, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_x, log10.(error_rates), log10.(x_error_MA_CA), color=:black, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MA_CB), color=:black, marker=:utriangle)

    # Naive compilation
    scatter!(f_x, log10.(error_rates), log10.(x_error_MB_CA), color=:red, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MB_CB), color=:red, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(f_x, log10.(error_rates), log10.(x_error_MS_CA), color=:blue, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MS_CB), color=:blue, marker=:utriangle)

    # Ancil reindex Compilation
    scatter!(f_x, log10.(error_rates), log10.(x_error_MC_CA), color=:green, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MC_CB), color=:green, marker=:utriangle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)
    # Z plot
    f_z = f[2,1]
    ax = f[2,1] = Axis(f_z, xlabel="p_mem",ylabel="Logical error rate",title=title*" Logical Z")
    lines!(f_z, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_z, log10.(error_rates), log10.(z_error_MA_CA), color=:black, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MA_CB), color=:black, marker=:utriangle)

    # Naive compilation
    scatter!(f_z, log10.(error_rates), log10.(z_error_MB_CA), color=:red, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MB_CB), color=:red, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(f_z, log10.(error_rates), log10.(z_error_MS_CA), color=:blue, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MS_CB), color=:blue, marker=:utriangle)
 
    # Ancil reindex Compilation
    scatter!(f_z, log10.(error_rates), log10.(z_error_MC_CA), color=:green, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MC_CB), color=:green, marker=:utriangle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)
    ################ Shor syndrome simulation ################
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    title = name*" Code - Shor Syndrome Circuit"

    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CA_shor)]
    z_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CA_shor)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CB_shor)]
    z_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CB_shor)]

    # Naive compilation and shuttle noise -> no gate noise and gate noise == init noise
    post_ec_error_rates_MB_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MB_CA_shor = [post_ec_error_rates_MB_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MB_CA_shor)]
    z_error_MB_CA_shor = [post_ec_error_rates_MB_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MB_CA_shor)]

    # Naive compilation and shuttle noise - gate noise == init noise 
    post_ec_error_rates_MB_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MB_CB_shor = [post_ec_error_rates_MB_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MB_CB_shor)]
    z_error_MB_CB_shor = [post_ec_error_rates_MB_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MB_CB_shor)]

    # Gate shuffled circuit 
    gate_shuffle_circ = CircuitCompilation2xn.gate_Shuffle!(scirc)

    # Gate shuffle - no gate noise
    post_ec_error_rates_MS_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, gate_shuffle_circ, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m,  nsamples=nsamples) for p in error_rates]
    x_error_MS_CA_shor = [post_ec_error_rates_MS_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MS_CA_shor)]
    z_error_MS_CA_shor = [post_ec_error_rates_MS_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MS_CA_shor)]

    # Gate shuffle- gate noise== init noise
    post_ec_error_rates_MS_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, gate_shuffle_circ, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MS_CB_shor = [post_ec_error_rates_MS_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MS_CB_shor)]
    z_error_MS_CB_shor = [post_ec_error_rates_MS_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MS_CB_shor)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    new_cat = CircuitCompilation2xn.perfect_reindex(cat, order)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, new_cat, new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA_shor = [post_ec_error_rates_MC_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MC_CA_shor)]
    z_error_MC_CA_shor = [post_ec_error_rates_MC_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MC_CA_shor)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, new_cat, new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MC_CB_shor = [post_ec_error_rates_MC_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MC_CB_shor)]
    z_error_MC_CB_shor = [post_ec_error_rates_MC_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MC_CB_shor)]

    shor_failed = false
    x_error_MD_CA_shor = []
    z_error_MD_CA_shor = []
    x_error_MD_CB_shor = []
    z_error_MD_CB_shor = []
    try
        # Special shor syndrome Compiled circuit
        shor_new_circuit, shor_order = CircuitCompilation2xn.ancil_reindex_pipeline_shor_syndrome(scirc)
        shor_cat = CircuitCompilation2xn.perfect_reindex(cat, shor_order)

        # TODO hack for working around gate commuting problem with Steane code
        if isa(code, QuantumClifford.ECC.Steane7)
            non_mz, mz = CircuitCompilation2xn.clifford_grouper(shor_new_circuit)
            batches = CircuitCompilation2xn.calculate_shifts(non_mz)
            reordered_batches = batches[[1,2,4,3,5,6]]
            new_non_mz = reduce(vcat, reordered_batches)
            shor_new_circuit = vcat(new_non_mz,mz)
        end

        # Special Shor circuit Compilation - no gate noise
        post_ec_error_rates_MD_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, shor_new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
        x_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MD_CA_shor)]
        z_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MD_CA_shor)]

        # Special Shor circuit Compilation - gate noise == init noise 
        post_ec_error_rates_MD_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, shor_new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
        x_error_MD_CB_shor = [post_ec_error_rates_MD_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MD_CB_shor)]
        z_error_MD_CB_shor = [post_ec_error_rates_MD_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MD_CB_shor)]
    catch e
        println("Step 5 was needed for SSSC")
        shor_failed = true
    end


    # Shor syndrome plots
    f_shor_x = f[1, 2]
    ax = f[1,2] = Axis(f_shor_x, xlabel="p_mem",title=title*" Logical X")
    lines!(f_shor_x, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MA_CA_shor), label="All to all (A2A) - no gate noise", color=:black, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MA_CB_shor), label="A2A - gate noise == after encoding error", color=:black, marker=:utriangle)

    # Naive compilation
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MB_CA_shor), label="Naive Compilation (NC) - no gate noise", color=:red, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MB_CB_shor), label="NC - gate noise == after encoding error", color=:red, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MS_CA_shor), label="Gate Shuffle (GS) - no gate noise", color=:blue, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MS_CB_shor), label="GS - gate noise == after encoding error", color=:blue, marker=:utriangle)

    # Ancil reindex Compilation
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MC_CA_shor), label="Ancil heuristic AH - no gate noise", color=:green, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MC_CB_shor), label="AH - gate noise == after encoding error", color=:green, marker=:utriangle)

    # Fancy Shor- specialized comilation
    if !shor_failed
        scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MD_CA_shor), label="Shor-syndrome Specialized comp (SSSC) - no gate noise", color=:orange, marker=:circle)
        scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MD_CB_shor), label="SSSC - gate noise == after encoding error", color=:orange, marker=:utriangle)
    end
    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)

    f_shor_z = f[2,2]
    ax = f[2,2] = Axis(f_shor_z, xlabel="p_mem",title=title*" Logical Z")
    lines!(f_shor_z, [-5, -0.5], [-5, -0.5], color=:gray)

    # All to all connectivty
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MA_CA_shor), color=:black, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MA_CB_shor), color=:black, marker=:utriangle)

    # Naive compilation
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MB_CA_shor), color=:red, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MB_CB_shor), color=:red, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MS_CA_shor), color=:blue, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MS_CB_shor), color=:blue, marker=:utriangle)

    # Ancil reindex Compilation
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MC_CA_shor), color=:green, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MC_CB_shor), color=:green, marker=:utriangle)

    # Fancy Shor- specialized comilation
    if !shor_failed
        scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MD_CA_shor), color=:orange, marker=:circle)
        scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MD_CB_shor), color=:orange, marker=:utriangle)
    end

    lines!(f_shor_z, [0,0], [0,0], label="All to all connectivty", color=:black)
    lines!(f_shor_z, [0,0], [0,0], label="Naive compilation", color=:red)
    lines!(f_shor_z, [0,0], [0,0], label="Gate shuffling", color=:blue) 
    lines!(f_shor_z, [0,0], [0,0], label="Ancil heuristic", color=:green)
    lines!(f_shor_z, [0,0], [0,0], label="Shor-syndrome specialized comp", color=:orange)

    scatter!(f_shor_z, [0,0], [0,0], label="Constant near-term noise parameters \n(shift error, decoherence, gate noise)", color=:gray, marker=:utriangle)
    scatter!(f_shor_z, [0,0], [0,0], label="Near-term noise parameters are \nmultiplied by 10*p_mem", color=:gray, marker=:circle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)
    f[1,3] = Legend(f, ax, "Compilation Style/Gate Noise")
    f[2,3] = Legend(f, ax, "Compilation Style/Gate Noise")

    return f 
end

######## Legacy code for plotting the LDPC codes given to me
function LDPC_plot(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_shift=0.0001, p_wait=1-exp(-14.5/28_000); name=string(typeof(code)))
    title = name*" Code - Naive Syndrome Circuit"
    checks = parity_checks(code)
    scirc, _ = naive_syndrome_circuit(checks)

    #ecirc = naive_encoding_circuit(code)
    ecirc = nothing

    nsamples = 10
    gate_fidelity = 0.9995
    m = 10 # improvment factor
    gate_noise = 1- gate_fidelity

    error_rates = exp10.(range(-4,-0.5,length=30))

    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA = [post_ec_error_rates_MA_CA[i][1] for i in eachindex(post_ec_error_rates_MA_CA)]
    z_error_MA_CA = [post_ec_error_rates_MA_CA[i][2] for i in eachindex(post_ec_error_rates_MA_CA)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB= [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB = [post_ec_error_rates_MA_CB[i][1] for i in eachindex(post_ec_error_rates_MA_CB)]
    z_error_MA_CB = [post_ec_error_rates_MA_CB[i][2] for i in eachindex(post_ec_error_rates_MA_CB)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA = [post_ec_error_rates_MC_CA[i][1] for i in eachindex(post_ec_error_rates_MC_CA)]
    z_error_MC_CA = [post_ec_error_rates_MC_CA[i][2] for i in eachindex(post_ec_error_rates_MC_CA)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB = [evaluate_code_decoder_naive_syndrome(checks, decoder, ecirc, new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MC_CB = [post_ec_error_rates_MC_CB[i][1] for i in eachindex(post_ec_error_rates_MC_CB)]
    z_error_MC_CB = [post_ec_error_rates_MC_CB[i][2] for i in eachindex(post_ec_error_rates_MC_CB)]

    f = Figure(size=(1500, 1500))
    # X plot
    f_x =  f[1,1]
    ax = f[1,1] = Axis(f_x, xlabel="p_mem = physical qubit error rate after encoding",ylabel="Logical error rate",title=title*" Logical X")
    lines!(f_x, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_x, log10.(error_rates), log10.(x_error_MA_CA), color=:black, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MA_CB), color=:black, marker=:utriangle)

    # Ancil reindex Compilation
    scatter!(f_x, log10.(error_rates), log10.(x_error_MC_CA), color=:green, marker=:circle)
    scatter!(f_x, log10.(error_rates), log10.(x_error_MC_CB), color=:green, marker=:utriangle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)
    # Z plot
    f_z = f[2,1]
    ax = f[2,1] = Axis(f_z, xlabel="p_mem",ylabel="Logical error rate",title=title*" Logical Z")
    lines!(f_z, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_z, log10.(error_rates), log10.(z_error_MA_CA), color=:black, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MA_CB), color=:black, marker=:utriangle)
 
    # Ancil reindex Compilation
    scatter!(f_z, log10.(error_rates), log10.(z_error_MC_CA), color=:green, marker=:circle)
    scatter!(f_z, log10.(error_rates), log10.(z_error_MC_CB), color=:green, marker=:utriangle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)

    ################ Shor syndrome simulation ################
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    title = name*" Code - Shor Syndrome Circuit"

    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CA_shor)]
    z_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CA_shor)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CB_shor)]
    z_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CB_shor)]

    # Special shor syndrome Compiled circuit
    shor_new_circuit, shor_order = CircuitCompilation2xn.ancil_reindex_pipeline_shor_syndrome(scirc)
    shor_cat = CircuitCompilation2xn.perfect_reindex(cat, shor_order)

    # TODO hack for working around gate commuting problem with Steane code
    if isa(code, QuantumClifford.ECC.Steane7)
        non_mz, mz = CircuitCompilation2xn.clifford_grouper(shor_new_circuit)
        batches = CircuitCompilation2xn.calculate_shifts(non_mz)
        reordered_batches = batches[[1,2,4,3,5,6]]
        new_non_mz = reduce(vcat, reordered_batches)
        shor_new_circuit = vcat(new_non_mz,mz)
    end

    # Special Shor circuit Compilation - no gate noise
    post_ec_error_rates_MD_CA_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, shor_new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MD_CA_shor)]
    z_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MD_CA_shor)]

    # Special Shor circuit Compilation - gate noise == init noise 
    post_ec_error_rates_MD_CB_shor = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, shor_new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MD_CB_shor = [post_ec_error_rates_MD_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MD_CB_shor)]
    z_error_MD_CB_shor = [post_ec_error_rates_MD_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MD_CB_shor)]

    # Shor syndrome plots
    f_shor_x = f[1, 2]
    ax = f[1,2] = Axis(f_shor_x, xlabel="p_mem",title=title*" Logical X")
    lines!(f_shor_x, [-5, -0.5], [-5, -0.5], label="single bit", color=:gray)

    # All to all connectivty
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MA_CA_shor), label="All to all (A2A) - no gate noise", color=:black, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MA_CB_shor), label="A2A - gate noise == after encoding error", color=:black, marker=:utriangle)

    # Fancy Shor- specialized comilation
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MD_CA_shor), label="Shor-syndrome Specialized comp (SSSC) - no gate noise", color=:orange, marker=:circle)
    scatter!(f_shor_x, log10.(error_rates), log10.(x_error_MD_CB_shor), label="SSSC - gate noise == after encoding error", color=:orange, marker=:utriangle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)

    f_shor_z = f[2,2]
    ax = f[2,2] = Axis(f_shor_z, xlabel="p_mem",title=title*" Logical Z")
    lines!(f_shor_z, [-5, -0.5], [-5, -0.5], color=:gray)

    # All to all connectivty
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MA_CA_shor), color=:black, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MA_CB_shor), color=:black, marker=:utriangle)

    # Fancy Shor- specialized comilation
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MD_CA_shor), color=:orange, marker=:circle)
    scatter!(f_shor_z, log10.(error_rates), log10.(z_error_MD_CB_shor), color=:orange, marker=:utriangle)

    lines!(f_shor_z, [0,0], [0,0], label="All to all connectivty", color=:black)
    lines!(f_shor_z, [0,0], [0,0], label="Naive compilation", color=:red)
    lines!(f_shor_z, [0,0], [0,0], label="Ancil heuristic", color=:green)
    lines!(f_shor_z, [0,0], [0,0], label="Shor-syndrome specialized comp", color=:orange)

    scatter!(f_shor_z, [0,0], [0,0], label="Constant near-term noise parameters \n(shift error, decoherence, gate noise)", color=:gray, marker=:utriangle)
    scatter!(f_shor_z, [0,0], [0,0], label="Near-term noise parameters are \nmultiplied by 10*p_mem", color=:gray, marker=:circle)

    xlims!(ax, high=-0.5, low=-4.0)
    ylims!(ax, high=-0.5, low=-4.0)
    f[1,3] = Legend(f, ax, "Compilation Style/Gate Noise")
    f[2,3] = Legend(f, ax, "Compilation Style/Gate Noise")

    return f 
end

# f_Steane = my_plot_both_synd(Steane7(), TableDecoder(Steane7()))
# f_Shor = my_plot_both_synd(Shor9(), TableDecoder(Shor9()))
# f_Cleve = my_plot_both_synd(Cleve8(), TableDecoder(Cleve8()))
# f_P5 = my_plot_both_synd(Perfect5(), TableDecoder(Perfect5()))

# f_x_Gottesman3 = my_plot_both_synd(Gottesman(3), TableDecoder(Gottesman(3)))

# f_t3 = my_plot_both_synd(Toric(3, 3), PyMatchingDecoder(Toric(3, 3)), name="Toric 3x3")
#f_t4 = my_plot_both_synd(Toric(4, 4), PyMatchingDecoder(Toric(4, 4)), name="Toric 4x4")
#f_t6 = my_plot_both_synd(Toric(6, 6), PyMatchingDecoder(Toric(6, 6)), name="Toric 6x6")
# f_t10 = the_plot_both_synd(Toric(10, 10), PyMatchingDecoder(Toric(10, 10)), name="Toric10")

H = LDPCDecoders.parity_check_matrix(n, wr, wc)
code = CSS(zeros(Bool,size(H)),H)
f_ldpc = LDPC_plot(code, PyBeliefPropDecoder(code))

# stab, Cx, Cz = CircuitCompilation2xn.getGoodLDPC(1)
# code = CSS(Cx, Cz)
# f_ldpc = LDPC_plot(code, PyBeliefPropDecoder(code))