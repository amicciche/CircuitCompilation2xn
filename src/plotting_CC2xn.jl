using CircuitCompilation2xn
using CircuitCompilation2xn: add_two_qubit_gate_noise, evaluate_code_decoder_shor_syndrome
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5
using CairoMakie
using QuantumClifford.ECC: AbstractSyndromeDecoder, TableDecoder, evaluate_decoder, Toric, PyMatchingDecoder
import PyQDecoders

"""This function is the asked for plot from Stefan in my DMs, 1/14/2024"""
function the_plot(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_shift=0.0001, p_wait=1-exp(-14.5/28_000); name=string(typeof(code)))
    title = name*" Circuit - Shor Syndrome Circuit"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)
    nsamples = 10_000

    error_rates = 0.000:0.00150:0.20
    error_rates = exp10.(range(-5,-1,length=40))

    # All to all connectivty - no gate noise 
    post_ec_error_rates_MA_CA = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, 0, 0, nsamples=nsamples) for p in error_rates]
    x_error_MA_CA = [post_ec_error_rates_MA_CA[i][1] for i in eachindex(post_ec_error_rates_MA_CA)]
    z_error_MA_CA = [post_ec_error_rates_MA_CA[i][2] for i in eachindex(post_ec_error_rates_MA_CA)]

    # All to all connectivty - gate noise == init noise 
    post_ec_error_rates_MA_CB= [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, add_two_qubit_gate_noise(scirc, p/10), p, 0, 0, nsamples=nsamples) for p in error_rates]
    x_error_MA_CB = [post_ec_error_rates_MA_CB[i][1] for i in eachindex(post_ec_error_rates_MA_CB)]
    z_error_MA_CB = [post_ec_error_rates_MA_CB[i][2] for i in eachindex(post_ec_error_rates_MA_CB)]

    # Naive compilation and shuttle noise -> no gate noise and gate noise == init noise
    post_ec_error_rates_MB_CA = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, scirc, p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MB_CA = [post_ec_error_rates_MB_CA[i][1] for i in eachindex(post_ec_error_rates_MB_CA)]
    z_error_MB_CA = [post_ec_error_rates_MB_CA[i][2] for i in eachindex(post_ec_error_rates_MB_CA)]

    # Naive compilation and shuttle noise - gate noise == init noise 
    post_ec_error_rates_MB_CB = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, add_two_qubit_gate_noise(scirc, p/10), p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MB_CB = [post_ec_error_rates_MB_CB[i][1] for i in eachindex(post_ec_error_rates_MB_CB)]
    z_error_MB_CB = [post_ec_error_rates_MB_CB[i][2] for i in eachindex(post_ec_error_rates_MB_CB)]

    # Gate shuffled circuit 
    non_mz, mz = CircuitCompilation2xn.clifford_grouper(scirc)
    CircuitCompilation2xn.gate_Shuffle!(non_mz)
    gate_shuffle_circ = vcat(non_mz, mz)

    # Gate shuffle - no gate noise
    post_ec_error_rates_MS_CA = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, gate_shuffle_circ, p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MS_CA = [post_ec_error_rates_MS_CA[i][1] for i in eachindex(post_ec_error_rates_MS_CA)]
    z_error_MS_CA = [post_ec_error_rates_MS_CA[i][2] for i in eachindex(post_ec_error_rates_MS_CA)]

    # Gate shuffle- gate noise== init noise
    post_ec_error_rates_MS_CB = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, cat, add_two_qubit_gate_noise(gate_shuffle_circ, p/10), p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MS_CB = [post_ec_error_rates_MS_CB[i][1] for i in eachindex(post_ec_error_rates_MS_CB)]
    z_error_MS_CB = [post_ec_error_rates_MS_CB[i][2] for i in eachindex(post_ec_error_rates_MS_CB)]

    # Circuit compilation
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    new_cat = CircuitCompilation2xn.perfect_reindex(cat, order)

    # Circuit comp - no gate noise
    post_ec_error_rates_MC_CA = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, new_cat, new_circuit, p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MC_CA = [post_ec_error_rates_MC_CA[i][1] for i in eachindex(post_ec_error_rates_MC_CA)]
    z_error_MC_CA = [post_ec_error_rates_MC_CA[i][2] for i in eachindex(post_ec_error_rates_MC_CA)]

    # Circuit Comp gate noise == init noise
    post_ec_error_rates_MC_CB = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, new_cat, add_two_qubit_gate_noise(new_circuit, p/10), p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
    x_error_MC_CB = [post_ec_error_rates_MC_CB[i][1] for i in eachindex(post_ec_error_rates_MC_CB)]
    z_error_MC_CB = [post_ec_error_rates_MC_CB[i][2] for i in eachindex(post_ec_error_rates_MC_CB)]

    shor_failed = false
    x_error_MD_CA = []
    z_error_MD_CA = []
    x_error_MD_CB = []
    z_error_MD_CB = []
    try
        # Special shor syndrome Compiled circuit
        shor_new_circuit, shor_order = CircuitCompilation2xn.ancil_reindex_pipeline_shor_syndrome(scirc)
        shor_cat = CircuitCompilation2xn.perfect_reindex(cat, shor_order)

        # Special Shor circuit Compilation - no gate noise
        post_ec_error_rates_MD_CA = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, shor_new_circuit, p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
        x_error_MD_CA = [post_ec_error_rates_MD_CA[i][1] for i in eachindex(post_ec_error_rates_MD_CA)]
        z_error_MD_CA = [post_ec_error_rates_MD_CA[i][2] for i in eachindex(post_ec_error_rates_MD_CA)]

        # Special Shor circuit Compilation - gate noise == init noise 
        post_ec_error_rates_MD_CB = [evaluate_code_decoder_shor_syndrome(checks, decoder, ecirc, shor_cat, add_two_qubit_gate_noise(shor_new_circuit, p/10), p, p_shift, p_wait, nsamples=nsamples) for p in error_rates]
        x_error_MD_CB = [post_ec_error_rates_MD_CB[i][1] for i in eachindex(post_ec_error_rates_MD_CB)]
        z_error_MD_CB = [post_ec_error_rates_MD_CB[i][2] for i in eachindex(post_ec_error_rates_MD_CB)]
    catch e
        println("Step 5 was needed for SSSC")
        shor_failed = true
    end

    # X plot
    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="physical qubit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical X")
    lines!([-5, 0], [-5, 0], label="single bit", color=:black)

    # All to all connectivty
    scatter!(log10.(error_rates), log10.(x_error_MA_CA), label="All to all (A2A) - no gate noise", color=:red, marker=:circle)
    scatter!(log10.(error_rates), log10.(x_error_MA_CB), label="A2A - gate noise == after encoding error", color=:blue, marker=:circle)

    # Naive compilation
    scatter!(log10.(error_rates), log10.(x_error_MB_CA), label="Naive Compilation (NC) - no gate noise", color=:red, marker=:utriangle)
    scatter!(log10.(error_rates), log10.(x_error_MB_CB), label="NC - gate noise == after encoding error", color=:blue, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(log10.(error_rates), log10.(x_error_MS_CA), label="Gate Shuffle (GS) - no gate noise", color=:red, marker=:star4)
    scatter!(log10.(error_rates), log10.(x_error_MS_CB), label="GS - gate noise == after encoding error", color=:blue, marker=:star4)

    # Ancil reindex Compilation
    scatter!(log10.(error_rates), log10.(x_error_MC_CA), label="Ancil heuristic AH - no gate noise", color=:red, marker=:rect)
    scatter!(log10.(error_rates), log10.(x_error_MC_CB), label="AH - gate noise == after encoding error", color=:blue, marker=:rect)

    # Fancy Shor- specialized comilation
    if !shor_failed
        scatter!(log10.(error_rates), log10.(x_error_MD_CA), label="Shor-syndrome Specialized comp (SSSC) - no gate noise", color=:red, marker=:star8)
        scatter!(log10.(error_rates), log10.(x_error_MD_CB), label="SSSC - gate noise == after encoding error", color=:blue, marker=:star8)
    end

    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    # Z plot
    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="physical qubit error rate - AKA error after encoding",ylabel="Logical error rate",title=title*" Logical Z")
    lines!([-5, 0], [-5, 0], label="single bit", color=:black)

    # All to all connectivty
    scatter!(log10.(error_rates), log10.(z_error_MA_CA), label="All to all (A2A) - no gate noise", color=:red, marker=:circle)
    scatter!(log10.(error_rates), log10.(z_error_MA_CB), label="A2A - gate noise == after encoding error", color=:blue, marker=:circle)

    # Naive compilation
    scatter!(log10.(error_rates), log10.(z_error_MB_CA), label="Naive Compilation (NC) - no gate noise", color=:red, marker=:utriangle)
    scatter!(log10.(error_rates), log10.(z_error_MB_CB), label="NC - gate noise == after encoding error", color=:blue, marker=:utriangle)

    # Gate shuffle compilation
    scatter!(log10.(error_rates), log10.(z_error_MS_CA), label="Gate Shuffle (GS) - no gate noise", color=:red, marker=:star4)
    scatter!(log10.(error_rates), log10.(z_error_MS_CB), label="GS - gate noise == after encoding error", color=:blue, marker=:star4)
 
    # Ancil reindex Compilation
    scatter!(log10.(error_rates), log10.(z_error_MC_CA), label="Ancil heuristic AH - no gate noise", color=:red, marker=:rect)
    scatter!(log10.(error_rates), log10.(z_error_MC_CB), label="AH - gate noise == after encoding error", color=:blue, marker=:rect)

    # Fancy Shor- specialized comilation
    if !shor_failed
        scatter!(log10.(error_rates), log10.(z_error_MD_CA), label="Shor-syndrome Specialized comp (SSSC) - no gate noise", color=:red, marker=:star8)
        scatter!(log10.(error_rates), log10.(z_error_MD_CB), label="SSSC - gate noise == after encoding error", color=:blue, marker=:star8)
    end
    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

# f_x_Steane, f_z_Steane = the_plot(Steane7(), TableDecoder(Steane7()))
# f_x_Shor, f_z_Shor = the_plot(Shor9(), TableDecoder(Shor9()))
# f_x_Cleve, f_z_Cleve = the_plot(Cleve8(), TableDecoder(Cleve8()))
# f_x_P5, f_z_P5 = the_plot(Perfect5(), TableDecoder(Perfect5()))

#f_x_t3, f_z_t3 = the_plot(Toric(3, 3), PyMatchingDecoder(Toric(3, 3)), name="Toric3")

# TODO L = 4,5,6  all needed step 5 for SSSC - so it caused an error as that isn't currently implemented
#f_x_t4, f_z_t4 = the_plot(Toric(4, 4), PyMatchingDecoder(Toric(4, 4)), name="Toric4")
#f_x_t5, f_z_t5 = the_plot(Toric(5, 5), PyMatchingDecoder(Toric(5, 5)), name="Toric5")
f_x_t6, f_z_t6 = the_plot(Toric(6, 6), PyMatchingDecoder(Toric(6, 6)), name="Toric6")
f_x_t10, f_z_t10 = the_plot(Toric(10, 10), PyMatchingDecoder(Toric(10, 10)), name="Toric10")