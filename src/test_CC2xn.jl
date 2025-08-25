using CircuitCompilation2xn
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k
using CairoMakie
using Random
using Statistics
using Distributions
using NPZ
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5
using LDPCDecoders
using CSV

function test_code(code)
    ecirc = naive_encoding_circuit(code)
    mcirc, _ = naive_syndrome_circuit(code)

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(mcirc)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code))

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
end

function test_full_reindex(code)
    ecirc = naive_encoding_circuit(code)
    mcirc, _ = naive_syndrome_circuit(code)

    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.reindex_by_dict(ecirc, data_order)

    diff = CircuitCompilation2xn.evaluate(mcirc, new_circuit, ecirc, code_n(code), code_s(code), code_s(code), new_ecirc, data_order)

    println("\nNumber of discrepancies between the reordered circuit and the original over all possible 1 qubit Pauli errors inserted right after the encoding circuit:")
    println(sum(diff))
    return new_ecirc, new_circuit
end

function test_full_reindex_plot(code, name=string(typeof(code)))
    ecirc = naive_encoding_circuit(code)
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(code)

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.reindex_by_dict(ecirc, data_order)

    error_rates = 0.000:0.0025:0.08
    H = parity_checks(code)
    dataQubits = size(H)[2]
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(H[:,parity_reindex], new_ecirc, new_circuit, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Data + Anc Reindexed "*name*" w/ Encoding Circuit")
    return f1
end

function no_encoding_plot(code, name=string(typeof(code)))
    checks = parity_checks(code)
    no_encoding_plot(checks, name)
end
function no_encoding_plot(checks::Stabilizer, name="")
    scirc, _ = naive_syndrome_circuit(checks)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(checks, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit - Syndrome Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder(checks, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit - Syndrome Circuit")
    return f1, f2
end

function encoding_plot(code, name=string(typeof(code)))
    checks = parity_checks(code)
    scirc, _ = naive_syndrome_circuit(code)
    ecirc = naive_encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(checks, ecirc, scirc, p) for p in error_rates]
    f1 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Original "*name*" Circuit w/ Encoding Circuit")

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc(checks, ecirc, new_circuit, p) for p in error_rates]
    f2 = CircuitCompilation2xn.plot_code_performance(error_rates, post_ec_error_rates,title="Reordered "*name*" Circuit w/ Encoding Circuit")
    return f1, f2
end

function pf_encoding_plot(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    pf_encoding_plot(checks, name)
end

function pf_encoding_plot(checks, name="")
    reduced_checks = copy(stabilizerview(MixedDestabilizer(checks)))
    scirc, _ = naive_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(reduced_checks)

    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit PF")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit PF")

    return f_x, f_z

    # Data-anc compile the circuit
    s, n = size(checks)
    k = n-s
    new_circuit, data_order = CircuitCompilation2xn.data_ancil_reindex(scirc, s+n)

    # Calculate locations of encoding qubits
    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end

    # Reindex encoding circuit
    new_ecirc = CircuitCompilation2xn.reindex_by_dict(ecirc, data_order)

    # Reindex the parity checks via checks[:,parity_reindex]
    dataQubits = n
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_pf(checks[:,parity_reindex], new_ecirc, new_circuit, p, 0, encoding_locs=encoding_locs) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    new_f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit PF")
    new_f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit PF")

    #return new_f_x, new_f_z
end

function encoding_plot_shifts(code, name=string(typeof(code)))
    scirc, _ = naive_syndrome_circuit(code)
    ecirc = naive_encoding_circuit(code)

    error_rates = 0.000:0.00150:0.08
    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, scirc, p, p/10) for p in error_rates]

    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    post_ec_error_rates_shifts = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, new_circuit, p, p/10) for p in error_rates]
    original = [CircuitCompilation2xn.evaluate_code_decoder_w_ecirc_shifts(parity_checks(code), ecirc, new_circuit, p, 0) for p in error_rates]
    plot = CircuitCompilation2xn.plot_code_performance_shift(error_rates, post_ec_error_rates, post_ec_error_rates_shifts,original, title=name*" Circuit w/ Encoding Circuit")
    return plot
end

function test_shor_circuit_reindexing(code, name=string(typeof(code)))
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit Shor_Syndrome")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit Shor_Syndrome")

    # anc compile the circuit
    new_circuit, order = CircuitCompilation2xn.ancil_reindex_pipeline(scirc)
    new_cat = CircuitCompilation2xn.reindex_by_dict(cat,order)

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    new_f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of AncReindex"*name*" Circuit ShorSynd")
    new_f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of AncReindex"*name*" Circuit ShorSynd")

    #return f_x, f_z
    return new_f_x, new_f_z
end

function test_shor_circuit_with_FTencode(code, name=string(typeof(code)))
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)

    error_rates = 0.000:0.0025:0.08

    post_ec_error_rates = [CircuitCompilation2xn.evaluate_code_FTencode_FTsynd_Krishna(checks, cat, scirc, p, 0) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit Shor_Syndrome")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit Shor_Syndrome")

    return f_x, f_z
end
#test_code(Steane7())
#test_code(Shor9())

#orig, new = no_encoding_plot(Cleve8())
#orig, new = encoding_plot(Shor9())
#orig, new = encoding_plot(Cleve8())

#f_x_Steane, f_z_Steane = pf_encoding_plot(Steane7())
#f_x_Shor, f_z_Shor = pf_encoding_plot(Shor9())
#f_x_Cleve, f_z_Cleve = pf_encoding_plot(Cleve8())

# f_x_Steane, f_z_Steane = CircuitCompilation2xn.vary_shift_errors_plot_pf(Steane7())
# f_x_Shor, f_z_Shor = CircuitCompilation2xn.vary_shift_errors_plot_pf(Shor9())
# f_x_Cleve, f_z_Cleve = CircuitCompilation2xn.vary_shift_errors_plot_pf(Cleve8())
# f_x_P5, f_z_P5 = CircuitCompilation2xn.vary_shift_errors_plot_pf(Perfect5())

#plot_3 = encoding_plot_shifts(Steane7())
#plot_3 = encoding_plot_shifts(Shor9())

#steane_e, steane_s = test_full_reindex(Steane7())
#shor_e, shor_s = test_full_reindex(Shor9())

#test_full_reindex_plot(Shor9())
#f_x_Steane, f_z_Steane = test_shor_circuit_reindexing(Steane7())
#f_x_Shor, f_z_Shor = test_shor_circuit_reindexing(Shor9())

 f_x_Steane, f_z_Steane = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Steane7())
 f_x_Shor, f_z_Shor = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Shor9())
 f_x_Cleve, f_z_Cleve = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Cleve8())
 f_x_P5, f_z_P5 = CircuitCompilation2xn.vary_shift_errors_plot_shor_syndrome(Perfect5())

######################### LDPC land #########################
#plot = CircuitCompilation2xn.plot_LDPC_shift_reduction_shiftPcheck()
#plot = CircuitCompilation2xn.plot_LDPC_shift_reduction_cooc()
######################### LDPC land #########################

#f_x_Steane, f_z_Steane = CircuitCompilation2xn.realistic_noise_logical_physical_error(Steane7())
#f_x_Shor, f_z_Shor = CircuitCompilation2xn.realistic_noise_logical_physical_error(Shor9())
#f_x_Cleve, f_z_Cleve = CircuitCompilation2xn.realistic_noise_logical_physical_error(Cle#ve8())
#f_x_P5, f_z_P5 = CircuitCompilation2xn.realistic_noise_logical_physical_error(Perfect5())
#f_x_P5, f_z_P5 = CircuitCompilation2xn.realistic_noise_vary_params(Perfect5())

function test_naive_refactor(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    test_naive_refactor(checks, name)
end
function test_naive_refactor(checks, name="")
    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [QuantumClifford.ECC.naive_error_correction_pipeline(checks, p) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Circuit PF")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Circuit PF")

    return f_x, f_z
end

function test_shor_refactor(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    test_shor_refactor(checks, name)
end
function test_shor_refactor(checks, name="")
    error_rates = 0.000:0.0025:0.08
    post_ec_error_rates = [QuantumClifford.ECC.shor_error_correction_pipeline(checks, p) for p in error_rates]
    x_error = [post_ec_error_rates[i][1] for i in eachindex(post_ec_error_rates)]
    z_error = [post_ec_error_rates[i][2] for i in eachindex(post_ec_error_rates)]

    f_x = CircuitCompilation2xn.plot_code_performance(error_rates, x_error,title="Logical X Error of "*name*" Shor Circuit PF")
    f_z = CircuitCompilation2xn.plot_code_performance(error_rates, z_error,title="Logical Z Error of "*name*" Shor Circuit PF")

    return f_x, f_z
end
# f_x_Steane, f_z_Steane = test_naive_refactor(Steane7())
# f_x_Shor, f_z_Shor = test_naive_refactor(Shor9())
# f_x_Cleve, f_z_Cleve = test_naive_refactor(Cleve8())
# f_x_P5, f_z_P5 = test_naive_refactor(Perfect5())

#f_x_Steane, f_z_Steane = test_shor_refactor(Steane7())
# f_x_Shor, f_z_Shor = test_shor_refactor(Shor9())
# f_x_Cleve, f_z_Cleve = test_shor_refactor(Cleve8())
# f_x_P5, f_z_P5 = test_shor_refactor(Perfect5())

function compilationNumbers_2xn(code::AbstractECC, shor_file=nothing, naive_file=nothing)
    # Shor circuits
    println("Shor syndrome extraction")
    x_checks = Stabilizer(BitArray(parity_matrix_x(code)), Matrix{Bool}(zeros(size(parity_matrix_x(code)))))
    z_checks = Stabilizer(Matrix{Bool}(zeros(size(parity_matrix_z(code)))), BitArray(parity_matrix_z(code)))

    println("X checks")
    cat, x_scirc, _ = QuantumClifford.ECC.shor_syndrome_circuit(x_checks);
    x_shor_numbers = CircuitCompilation2xn.shorNumbers(x_scirc)

    println("\nZ checks")
    cat, z_scirc, _ = QuantumClifford.ECC.shor_syndrome_circuit(z_checks);
    z_shor_numbers = CircuitCompilation2xn.shorNumbers(z_scirc)

    println("\n\nNaive syndrome extraction")

    println("X checks")
    naive_x_scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(x_checks);
    x_naive_numbers = CircuitCompilation2xn.comp_numbers(naive_x_scirc, code_n(code)+code_s(code))

    println("\nZ checks")
    naive_z_scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(z_checks);
    z_naive_numbers = CircuitCompilation2xn.comp_numbers(naive_z_scirc, code_n(code)+code_s(code))


    if !isnothing(shor_file)
        shor_row_x = (Code=string(code)*"_x", Uncompiled=x_shor_numbers[1], GateShuffled=x_shor_numbers[2], AHR=x_shor_numbers[3], SSSC=x_shor_numbers[4],AllowingBlanks=(x_shor_numbers[5],x_shor_numbers[6]))
        shor_row_z = (Code=string(code)*"_z", Uncompiled=z_shor_numbers[1], GateShuffled=z_shor_numbers[2], AHR=z_shor_numbers[3], SSSC=z_shor_numbers[4],AllowingBlanks=(z_shor_numbers[5],z_shor_numbers[6]))

        CSV.write(shor_file, [shor_row_x], append=true, writeheader=false)
        CSV.write(shor_file, [shor_row_z], append=true, writeheader=false)
    end

    if !isnothing(naive_file)
        naive_row_x = (Code=string(code)*"_x", Uncompiled=x_naive_numbers[1], GateShuffled=x_naive_numbers[2], AHR=x_naive_numbers[3], DataReindexing=x_naive_numbers[4])
        naive_row_z = (Code=string(code)*"_z", Uncompiled=z_naive_numbers[1], GateShuffled=z_naive_numbers[2], AHR=z_naive_numbers[3], DataReindexing=z_naive_numbers[4])

        CSV.write(naive_file, [naive_row_x], append=true, writeheader=false)
        CSV.write(naive_file, [naive_row_z], append=true, writeheader=false)
    end
end
#compilationNumbers_2xn(code, "./CircuitCompilation2xn/shor_compilation_numbers_data.csv",  "./CircuitCompilation2xn/naive_compilation_numbers_data.csv")