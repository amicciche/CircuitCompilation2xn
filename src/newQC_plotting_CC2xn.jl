using CircuitCompilation2xn
using CircuitCompilation2xn: add_two_qubit_gate_noise, plot_code_performance, fault_tolerant_encoding
using QuantumClifford
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, parity_checks, code_s, code_n, code_k, faults_matrix
using QuantumClifford.ECC: naive_encoding_circuit, Cleve8, AbstractECC, Perfect5, AbstractSyndromeDecoder, TableDecoder, evaluate_decoder
using CairoMakie

function naive_pipeline(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, scirc=nothing, ecirc=nothing)
    return naive_pipeline(parity_checks(code), decoder, p_mem, nsamples=nsamples, scirc=scirc, ecirc=ecirc)
end
function naive_pipeline(checks::Stabilizer, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, scirc=nothing, ecirc=nothing)
    if isnothing(scirc)
        scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(checks)
    end
    O = faults_matrix(checks)

    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    _, _, r = canonicalize!(Base.copy(checks), ranks=true)
    k = n - r
    pre_X = [sHadamard(i) for i in n-k+1:n]

    md = MixedDestabilizer(checks)
    logview_Z = logicalzview(md)
    logcirc_Z, numLogBits_Z, _ = naive_syndrome_circuit(logview_Z)
    @assert numLogBits_Z == k

    logview_X = logicalxview(md)
    logcirc_X, numLogBits_X, _ = naive_syndrome_circuit(logview_X)
    @assert numLogBits_X == k

    syndrome_bits = 1:s
    logical_bits = s+1:s+k

    x_sub_matrix = O[1:k, :]
    z_sub_matrix = O[k+1:2k, :]

    errors = [PauliError(i,p_mem) for i in 1:n]

    # Z logic circuit
    for gate in logcirc_Z
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_Z, sMRZ(gate.qubit+s, gate.bit+s))
        else
            push!(circuit_Z, type(gate.q1, gate.q2+s))
        end
    end

   # X logic circuit
    for gate in logcirc_X
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_X, sMRZ(gate.qubit+s, gate.bit+s))
        else
            push!(circuit_X, type(gate.q1, gate.q2+s))
        end
    end

    if isnothing(ecirc)
        ecirc_z = fault_tolerant_encoding(circuit_Z) # double syndrome encoding
        ecirc_x = fault_tolerant_encoding(circuit_X) 
    else
        ecirc_z = ecirc
        ecirc_x = ecirc
    end

    full_x_circ = vcat(pre_X, ecirc_x, errors,  circuit_X)
    full_z_circ = vcat(ecirc_z, errors, circuit_Z)

    x_error = evaluate_decoder(decoder, nsamples, full_x_circ, syndrome_bits, logical_bits, x_sub_matrix)
    z_error = evaluate_decoder(decoder, nsamples, full_z_circ, syndrome_bits, logical_bits, z_sub_matrix)

    return x_error, z_error
end

function shor_pipeline(code::AbstractECC, decoder::AbstractSyndromeDecoder, p_mem, nsamples=10_000, cat=nothing, scirc=nothing, ecirc=nothing)
    return shor_pipeline(parity_checks(code), decoder, p_mem, nsamples=nsamples, cat=cat, scirc=scirc, ecirc=ecirc)
end
function shor_pipeline(checks::Stabilizer, decoder::AbstractSyndromeDecoder, p_mem; nsamples=10_000, cat=nothing, scirc=nothing, ecirc=nothing)
    if isnothing(scirc) || isnothing(cat)
        cat, scirc, _ = shor_syndrome_circuit(checks)
    end

    O = faults_matrix(checks)
    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    _, _, r = canonicalize!(Base.copy(checks), ranks=true)
    k = n - r
    pre_X = [sHadamard(i) for i in n-k+1:n]

    anc_qubits = 0
    for pauli in checks
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end
    regbits = anc_qubits + s

    md = MixedDestabilizer(checks)
    logview_Z = logicalzview(md)
    logcirc_Z, _ = naive_syndrome_circuit(logview_Z)

    logview_X = logicalxview(md)
    logcirc_X, _ = naive_syndrome_circuit(logview_X)

    # Z logic circuit
    for gate in logcirc_Z
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_Z, sMRZ(gate.qubit+anc_qubits, gate.bit+regbits))
        else
            push!(circuit_Z, type(gate.q1, gate.q2+anc_qubits))
        end
    end

   # X logic circuit
    for gate in logcirc_X
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_X, sMRZ(gate.qubit+anc_qubits, gate.bit+regbits))
        else
            push!(circuit_X, type(gate.q1, gate.q2+anc_qubits))
        end
    end

    errors = [PauliError(i,p_mem) for i in 1:n]
    if isnothing(ecirc)
        ecirc_z = fault_tolerant_encoding(vcat(cat,circuit_Z)) # double syndrome encoding
        ecirc_x = fault_tolerant_encoding(vcat(cat,circuit_X))
        full_z_circ = vcat(ecirc_z, errors, circuit_Z)
        full_x_circ = vcat(pre_X, ecirc_x, errors, circuit_X)
    else
        full_z_circ = vcat(ecirc, errors, cat, circuit_Z)
        full_x_circ = vcat(pre_X, ecirc, errors, cat, circuit_X)
    end

    syndrome_bits = anc_qubits+1:regbits
    logical_bits = regbits+1:regbits+k

    x_sub_matrix = O[1:k, :]
    z_sub_matrix = O[k+1:2k, :]

    x_error = evaluate_decoder(decoder, nsamples, full_x_circ, syndrome_bits, logical_bits, x_sub_matrix)
    z_error = evaluate_decoder(decoder, nsamples, full_z_circ, syndrome_bits, logical_bits, z_sub_matrix)

    return x_error, z_error
end

function the_plot(code::AbstractECC, decoder::AbstractSyndromeDecoder=TableDecoder(code); name=string(typeof(code)))
    error_rates = 0.000:0.0050:0.15
    post_ec = [shor_pipeline(code, decoder, p) for p in error_rates]
    x_results = [post_ec[i][1] for i in eachindex(post_ec)]
    z_results = [post_ec[i][2] for i in eachindex(post_ec)]
    f_x = plot_code_performance(error_rates, x_results, title= name*" Logical X error")
    f_z = plot_code_performance(error_rates, z_results, title= name*" Logical Z error")

    return f_x, f_z
end

f_x_Steane, f_z_Steane = the_plot(Steane7())
f_x_Shor, f_z_Shor = the_plot(Shor9())
f_x_Cleve, f_z_Cleve = the_plot(Cleve8())
f_x_P5, f_z_P5 = the_plot(Perfect5())

f_x_t3, f_z_t3 = the_plot(Toric(3, 3), PyMatchingDecoder(Toric(3, 3)), name="Toric3")
f_x_t6, f_z_t6 = the_plot(Toric(6, 6), PyMatchingDecoder(Toric(6, 6)), name="Toric6")
f_x_t10, f_z_t10 = the_plot(Toric(10, 10), PyMatchingDecoder(Toric(10, 10)), name="Toric10")