# File for running misc tests
import PyQDecoders
using CircuitCompilation2xn
using CircuitCompilation2xn: add_two_qubit_gate_noise, plot_code_performance, fault_tolerant_encoding, shor_pipeline, naive_pipeline, bootleg_CSS, CSS_naive_error_correction_pipeline, CSS_shor_error_correction_pipeline, CSS_evaluate_code_decoder_naive_syndrome, CSS_evaluate_code_decoder_shor_syndrome
using QuantumClifford
using QuantumClifford.ECC
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k, faults_matrix, Gottesman, CSS, PyBeliefPropOSDecoder, PyBeliefPropDecoder
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5, AbstractSyndromeDecoder, TableDecoder, evaluate_decoder, Toric, PyMatchingDecoder
using CairoMakie

function simple_log_log_plot(phys_errors, results, title="")
    f = Figure(size=(750, 750))

    f_x =  f[1,1]
    ax = f[1,1] = Axis(f_x, xlabel="p_mem = physical qubit error rate after encoding",ylabel="Logical error rate",title=title)
    lines!(f_x, [-6, -0.5], [-6, -0.5], label="single bit", color=:gray)

    scatter!(f_x, log10.(error_rates), log10.(results), color=:black, marker=:circle)

    f
end

function the_plot(checks::Stabilizer, decoder::AbstractSyndromeDecoder=TableDecoder(code); name=string(typeof(code)))
    error_rates = 0.000:0.0050:0.1
    error_rates = exp10.(range(-6,-1,length=30))
    post_ec = [naive_pipeline(checks, decoder, p, nsamples=5_000) for p in error_rates]
    x_results = [post_ec[i][1] for i in eachindex(post_ec)]
    z_results = [post_ec[i][2] for i in eachindex(post_ec)]
    f_x = simple_log_log_plot(error_rates, x_results, name*" Logical X error")
    f_z = simple_log_log_plot(error_rates, z_results, name*" Logical Z error")

    return f_x, f_z
end
function the_plot(code::AbstractECC, decoder::AbstractSyndromeDecoder=TableDecoder(code); name=string(typeof(code)))
    error_rates = 0.000:0.0050:0.1
    error_rates = exp10.(range(-6,-1,length=30))
    post_ec = [naive_pipeline(code, decoder, p, nsamples=5_000) for p in error_rates]
    x_results = [post_ec[i][1] for i in eachindex(post_ec)]
    z_results = [post_ec[i][2] for i in eachindex(post_ec)]
    f_x = simple_log_log_plot(error_rates, x_results, name*" Logical X error")
    f_z = simple_log_log_plot(error_rates, z_results, name*" Logical Z error")

    return f_x, f_z
end

function CSS_my_plot_both_synd(code::bootleg_CSS, p_shift=0.0001, p_wait=1-exp(-14.5/28_000); name=string(typeof(code)))
    title = name*" Code - Naive Syndrome Circuit"
    checks = code.tableau
    scirc, _ = naive_syndrome_circuit(checks)
    ecirc = nothing
    nsamples = 1_000
    gate_fidelity = 0.9995
    m = 10 # improvment factor
    gate_noise = 1- gate_fidelity

    error_rates = exp10.(range(-5,-1,length=25))

    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA = [CSS_evaluate_code_decoder_naive_syndrome(code, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA = [post_ec_error_rates_MA_CA[i][1] for i in eachindex(post_ec_error_rates_MA_CA)]
    z_error_MA_CA = [post_ec_error_rates_MA_CA[i][2] for i in eachindex(post_ec_error_rates_MA_CA)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB= [CSS_evaluate_code_decoder_naive_syndrome(code, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB = [post_ec_error_rates_MA_CB[i][1] for i in eachindex(post_ec_error_rates_MA_CB)]
    z_error_MA_CB = [post_ec_error_rates_MA_CB[i][2] for i in eachindex(post_ec_error_rates_MA_CB)]

    # Naive compilation and shuttle noise -> no gate noise and gate noise == init noise
    post_ec_error_rates_MB_CA = [CSS_evaluate_code_decoder_naive_syndrome(code, scirc, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MB_CA = [post_ec_error_rates_MB_CA[i][1] for i in eachindex(post_ec_error_rates_MB_CA)]
    z_error_MB_CA = [post_ec_error_rates_MB_CA[i][2] for i in eachindex(post_ec_error_rates_MB_CA)]

    # Naive compilation and shuttle noise - gate noise == init noise 
    post_ec_error_rates_MB_CB = [CSS_evaluate_code_decoder_naive_syndrome(code, scirc, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MB_CB = [post_ec_error_rates_MB_CB[i][1] for i in eachindex(post_ec_error_rates_MB_CB)]
    z_error_MB_CB = [post_ec_error_rates_MB_CB[i][2] for i in eachindex(post_ec_error_rates_MB_CB)]

    # Gate shuffled circuit 
    gate_shuffle_circ = CircuitCompilation2xn.gate_Shuffle!(scirc)

    # Gate shuffle - no gate noise
    post_ec_error_rates_MS_CA = [CSS_evaluate_code_decoder_naive_syndrome(code, gate_shuffle_circ, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MS_CA = [post_ec_error_rates_MS_CA[i][1] for i in eachindex(post_ec_error_rates_MS_CA)]
    z_error_MS_CA = [post_ec_error_rates_MS_CA[i][2] for i in eachindex(post_ec_error_rates_MS_CA)]

    # Gate shuffle- gate noise== init noise
    post_ec_error_rates_MS_CB = [CSS_evaluate_code_decoder_naive_syndrome(code, gate_shuffle_circ, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MS_CB = [post_ec_error_rates_MS_CB[i][1] for i in eachindex(post_ec_error_rates_MS_CB)]
    z_error_MS_CB = [post_ec_error_rates_MS_CB[i][2] for i in eachindex(post_ec_error_rates_MS_CB)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA = [CSS_evaluate_code_decoder_naive_syndrome(code, new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA = [post_ec_error_rates_MC_CA[i][1] for i in eachindex(post_ec_error_rates_MC_CA)]
    z_error_MC_CA = [post_ec_error_rates_MC_CA[i][2] for i in eachindex(post_ec_error_rates_MC_CA)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB = [CSS_evaluate_code_decoder_naive_syndrome(code, new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
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
    post_ec_error_rates_MA_CA_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, scirc, p, 0, 0, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CA_shor)]
    z_error_MA_CA_shor = [post_ec_error_rates_MA_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CA_shor)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, scirc, p, 0, 0, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MA_CB_shor)]
    z_error_MA_CB_shor = [post_ec_error_rates_MA_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MA_CB_shor)]

    # Naive compilation and shuttle noise -> no gate noise and gate noise == init noise
    post_ec_error_rates_MB_CA_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, scirc, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MB_CA_shor = [post_ec_error_rates_MB_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MB_CA_shor)]
    z_error_MB_CA_shor = [post_ec_error_rates_MB_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MB_CA_shor)]

    # Naive compilation and shuttle noise - gate noise == init noise 
    post_ec_error_rates_MB_CB_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, scirc, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MB_CB_shor = [post_ec_error_rates_MB_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MB_CB_shor)]
    z_error_MB_CB_shor = [post_ec_error_rates_MB_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MB_CB_shor)]

    # Gate shuffled circuit 
    gate_shuffle_circ = CircuitCompilation2xn.gate_Shuffle!(scirc)

    # Gate shuffle - no gate noise
    post_ec_error_rates_MS_CA_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, gate_shuffle_circ, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m,  nsamples=nsamples) for p in error_rates]
    x_error_MS_CA_shor = [post_ec_error_rates_MS_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MS_CA_shor)]
    z_error_MS_CA_shor = [post_ec_error_rates_MS_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MS_CA_shor)]

    # Gate shuffle- gate noise== init noise
    post_ec_error_rates_MS_CB_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, cat, gate_shuffle_circ, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
    x_error_MS_CB_shor = [post_ec_error_rates_MS_CB_shor[i][1] for i in eachindex(post_ec_error_rates_MS_CB_shor)]
    z_error_MS_CB_shor = [post_ec_error_rates_MS_CB_shor[i][2] for i in eachindex(post_ec_error_rates_MS_CB_shor)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    new_cat = CircuitCompilation2xn.reindex_by_dict(cat, order)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, new_cat, new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA_shor = [post_ec_error_rates_MC_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MC_CA_shor)]
    z_error_MC_CA_shor = [post_ec_error_rates_MC_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MC_CA_shor)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, new_cat, new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
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
        shor_cat = CircuitCompilation2xn.reindex_by_dict(cat, shor_order)

        # Special Shor circuit Compilation - no gate noise
        post_ec_error_rates_MD_CA_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, shor_cat, shor_new_circuit, p, p_shift*p*m, p_wait*p*m, gate_noise*p*m, nsamples=nsamples) for p in error_rates]
        x_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][1] for i in eachindex(post_ec_error_rates_MD_CA_shor)]
        z_error_MD_CA_shor = [post_ec_error_rates_MD_CA_shor[i][2] for i in eachindex(post_ec_error_rates_MD_CA_shor)]

        # Special Shor circuit Compilation - gate noise == init noise 
        post_ec_error_rates_MD_CB_shor = [CSS_evaluate_code_decoder_shor_syndrome(code, shor_cat, shor_new_circuit, p, p_shift, p_wait, gate_noise, nsamples=nsamples) for p in error_rates]
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

# stab, Cx, Cz = CircuitCompilation2xn.getGoodLDPC(1)
# code = bootleg_CSS(stab, Cx, Cz)
# f_ldpc1 = CSS_my_plot_both_synd(code, name="LDPC1")

#_x_Steane, f_z_Steane = the_plot(Steane7())
#f_x_Shor, f_z_Shor = the_plot(Shor9())
#f_x_Cleve, f_z_Cleve = the_plot(Cleve8())
#f_x_P5, f_z_P5 = the_plot(Perfect5())

#f_x_Gottesman3, f_z_Gottesman3 = the_plot(Gottesman(3))

#f_x_t3, f_z_t3 = the_plot(Toric(3, 3), PyMatchingDecoder(Toric(3, 3)), name="Toric3")
# f_x_t6, f_z_t6 = the_plot(Toric(6, 6), PyMatchingDecoder(Toric(6, 6)), name="Toric6")
# f_x_t10, f_z_t10 = the_plot(Toric(10, 10), PyMatchingDecoder(Toric(10, 10)), name="Toric10")

stab, Cx, Cz = CircuitCompilation2xn.getGoodLDPC(1)
ldpc1 = CSS(Cx,Cz)
#f_ldpc1_x, f_ldpc1_z = the_plot(ldpc1, TableDecoder(ldpc1), name="LDPC1")

make_decoder_figure(mem_errors, results, "LDPC plot1 - CommutationCheck - TableDecoder")

# error_rates = exp10.(range(-6,-2,length=20))
# post_ec = [CSS_shor_error_correction_pipeline(code, p, nframes=25_000) for p in error_rates]
# x_results = [post_ec[i][1] for i in eachindex(post_ec)]
# z_results = [post_ec[i][2] for i in eachindex(post_ec)]
# name = "LDPC1"
# f_x = simple_log_log_plot(error_rates, x_results, name*" Logical X error")
# f_z = simple_log_log_plot(error_rates, z_results, name*" Logical Z error")