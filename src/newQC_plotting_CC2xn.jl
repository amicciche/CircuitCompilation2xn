using CircuitCompilation2xn
using CircuitCompilation2xn: add_two_qubit_gate_noise, plot_code_performance, fault_tolerant_encoding, shor_pipeline, naive_pipeline
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k, faults_matrix
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5, AbstractSyndromeDecoder, TableDecoder, evaluate_decoder
using CairoMakie

function the_plot(code::AbstractECC, decoder::AbstractSyndromeDecoder=TableDecoder(code); name=string(typeof(code)))
    error_rates = 0.000:0.0050:0.15
    post_ec = [naive_pipeline(code, decoder, p) for p in error_rates]
    x_results = [post_ec[i][1] for i in eachindex(post_ec)]
    z_results = [post_ec[i][2] for i in eachindex(post_ec)]
    f_x = plot_code_performance(error_rates, x_results, title= name*" Logical X error")
    f_z = plot_code_performance(error_rates, z_results, title= name*" Logical Z error")

    return f_x, f_z
end

# f_x_Steane, f_z_Steane = the_plot(Steane7())
# f_x_Shor, f_z_Shor = the_plot(Shor9())
# f_x_Cleve, f_z_Cleve = the_plot(Cleve8())
# f_x_P5, f_z_P5 = the_plot(Perfect5())

# f_x_t3, f_z_t3 = the_plot(Toric(3, 3), PyMatchingDecoder(Toric(3, 3)), name="Toric3")
# f_x_t6, f_z_t6 = the_plot(Toric(6, 6), PyMatchingDecoder(Toric(6, 6)), name="Toric6")
# f_x_t10, f_z_t10 = the_plot(Toric(10, 10), PyMatchingDecoder(Toric(10, 10)), name="Toric10")