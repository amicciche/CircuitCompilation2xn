using CircuitCompilation2xn
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, encoding_circuit, parity_checks, code_s, code_n
using Quantikz
using CairoMakie

function test_code(code)
    ecirc = encoding_circuit(code)
    mcirc = naive_syndrome_circuit(code)

    new_circuit = CircuitCompilation2xn.test(mcirc)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code))

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
end

function steane_plots()
    code = Steane7()
    scirc = naive_syndrome_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_noisy_circuit(parity_checks(code), scirc, p, 0) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original Steane7 Circuit")

    new_circuit = CircuitCompilation2xn.test(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_noisy_circuit(parity_checks(code), new_circuit, p, 0) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered Steane7 Circuit")
    return f1, f2
end

#println("\n######################### Steane7 #########################")
#test_code(Steane7())

#println("\n######################### Shor9 #########################")
#test_code(Shor9())

println("\n######################### Steane7 Plots #########################")
orig, new = steane_plots()