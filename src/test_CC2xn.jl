using CircuitCompilation2xn
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, encoding_circuit, parity_checks, code_s, code_n, code_k
using Quantikz
using CairoMakie

function test_code(code)
    ecirc = encoding_circuit(code)
    mcirc = naive_syndrome_circuit(code)

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(mcirc)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code))

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
end

function test_full_reindex(code)
    ecirc = encoding_circuit(code)
    mcirc = naive_syndrome_circuit(code)

    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.encoding_reindex(ecirc, data_order)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code), new_ecirc, data_order)

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
    return new_ecirc, new_circuit
end

# TODO NEED TO REINDEX THE PARITY checks
function test_full_reindex_plot(code, name=string(typeof(code)))
    ecirc = encoding_circuit(code)
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.encoding_reindex(ecirc, data_order)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code)[:,data_order], new_ecirc, new_circuit, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Data + Anc Reindexed "*name*" w/ Encoding Circuit")
    return f1
end

function no_encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit - Syndrome Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(parity_checks(code), new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit - Syndrome Circuit")
    return f1, f2
end

function encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code), ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(parity_checks(code), ecirc, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit")
    return f1, f2
end

function pf_encoding_plot(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(parity_checks(code), ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit PF")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(parity_checks(code), ecirc, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit PF")
    return f1, f2
end

function encoding_plot_shifts(code, name=string(typeof(code)))
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)

    error_rates = 0.000:0.00150:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, scirc, p, p/10) for p in error_rates]

    new_circuit, order = CircuitCompilation2xn.ancil_reindex(scirc)
    post_ec_error_rates_shifts = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, new_circuit, p, p/10) for p in error_rates]
    original = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(parity_checks(code), ecirc, new_circuit, p) for p in error_rates]
    plot = CircuitCompilation2xn.plot_code_performance_shift(error_rates, post_ec_error_rates, post_ec_error_rates_shifts,original, title=name*" Circuit w/ Encoding Circuit")
    return plot
end

#println("\n######################### Steane7 #########################")
#test_code(Steane7())

#println("\n######################### Shor9 #########################")
#test_code(Shor9())

#println("\n######################### Shor9 Plots #########################")

#orig, new = encoding_plot(Steane7())
#orig, new = encoding_plot(Shor9())

#orig, new = pf_encoding_plot(Steane7())
#orig, new = pf_encoding_plot(Shor9())

#plot_3 = encoding_plot_shifts(Steane7())
#plot_3 = encoding_plot_shifts(Shor9())

plot = CircuitCompilation2xn.vary_shift_errors_plot(Steane7())
#plot = CircuitCompilation2xn.vary_shift_errors_plot(Shor9())

#steane_e, steane_s = test_full_reindex(Steane7())
#shor_e, shor_s = test_full_reindex(Shor9())

#test_full_reindex_plot(Shor9())