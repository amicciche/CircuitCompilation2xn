"""PauliFrame version of [`evaluate_code_decoder_w_ecirc`](@ref) and [`evaluate_code_decoder_w_ecirc_shifts`](@ref).
If no p_shift is provided, this runs as if it there were no shift errors."""
function evaluate_code_decoder_w_ecirc_pf(checks::Stabilizer, ecirc, scirc, p_init, p_shift=0 ; nframes=10_000, encoding_locs=nothing)   
    lookup_table = create_lookup_table(checks)
    O = faults_matrix(checks)
    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    k = n-s

    if isnothing(encoding_locs)
        pre_X = [sHadamard(i) for i in n-k+1:n]
    else
        pre_X = [sHadamard(i) for i in encoding_locs]
    end     
    
    if p_shift != 0       
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        circuit_X = []
        circuit_Z = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                append!(circuit_X, [PauliError(i,p_shift) for i in n+1:n+s])
                append!(circuit_Z, [PauliError(i,p_shift) for i in n+1:n+s])
            end
            append!(circuit_Z, subcircuit)
            append!(circuit_X, subcircuit)
            first_shift = false
        end
        append!(circuit_X, mz)
        append!(circuit_Z, mz)
    end

    md = MixedDestabilizer(checks)
    logview_Z = [ logicalzview(md);]
    logcirc_Z, numLogBits, _ = naive_syndrome_circuit(logview_Z) # numLogBits shoudl equal k

    logview_X = [ logicalxview(md);]
    logcirc_X, _ = naive_syndrome_circuit(logview_X)
    
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

    # Z simulation
    errors = [PauliError(i,p_init) for i in 1:n]
    fullcircuit_Z = vcat(ecirc, errors, circuit_Z)

    frames = PauliFrame(nframes, n+s+k, s+k)
    pftrajectories(frames, fullcircuit_Z)
    syndromes = pfmeasurements(frames)[:, 1:s]
    logicalSyndromes = pfmeasurements(frames)[:, s+1: s+k] 

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_Z = (O * stab_to_gf2(guess))[k+1:2k]
            if result_Z == logicalSyndromes[i,:]
                decoded += 1
            end
        end
    end
    z_error = 1 - decoded / nframes
    
    # X simulation
    fullcircuit_X = vcat(pre_X, ecirc, errors, circuit_X)
    frames = PauliFrame(nframes, n+s+k, s+k)
    pftrajectories(frames, fullcircuit_X)
    syndromes = pfmeasurements(frames)[:, 1:s]
    logicalSyndromes = pfmeasurements(frames)[:, s+1: s+k]

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_X = (O * stab_to_gf2(guess))[1:k]
            if result_X == logicalSyndromes[i, :]
                decoded += 1
            end
        end
    end
    x_error = 1 - decoded / nframes

    return x_error, z_error
end

"""Evaluates a decoder for shor style syndrome circuit"""
function evaluate_code_decoder_shor_syndrome(checks::Stabilizer, ecirc, cat, scirc, p_init, p_shift=0 ; nframes=10_000, encoding_locs=nothing)   
    lookup_table = create_lookup_table(checks)
    O = faults_matrix(checks)
    circuit_Z = Base.copy(scirc)
    circuit_X = Base.copy(scirc)

    s, n = size(checks)
    k = n-s

    if isnothing(encoding_locs)
        pre_X = [sHadamard(i) for i in n-k+1:n]
    else
        pre_X = [sHadamard(i) for i in encoding_locs]
    end 
        
    anc_qubits = 0 
    for pauli in checks
        anc_qubits += mapreduce(count_ones,+, xview(pauli) .| zview(pauli))
    end

    regbits = anc_qubits + s

    if p_shift != 0       
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        circuit_X = []
        circuit_Z = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                append!(circuit_X, [PauliError(i,p_shift) for i in n+1:n+anc_qubits])
                append!(circuit_Z, [PauliError(i,p_shift) for i in n+1:n+anc_qubits])
            end
            append!(circuit_Z, subcircuit)
            append!(circuit_X, subcircuit)
            first_shift = false
        end
        append!(circuit_X, mz)
        append!(circuit_Z, mz)
    end

    # TODO make these logical circuits also shor style?
    md = MixedDestabilizer(checks)
    logview_Z = [ logicalzview(md);]
    logcirc_Z, _ = naive_syndrome_circuit(logview_Z)

    logview_X = [ logicalxview(md);]
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

    # Z simulation
    errors = [PauliError(i,p_init) for i in 1:n]
    fullcircuit_Z = vcat(ecirc, errors, cat, circuit_Z)

    frames = PauliFrame(nframes, n+anc_qubits+k, regbits+k)
    pftrajectories(frames, fullcircuit_Z)
    syndromes = pfmeasurements(frames)[:, anc_qubits+1:regbits]
    logicalSyndromes = pfmeasurements(frames)[:, regbits+1:regbits+k]

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_Z = (O * stab_to_gf2(guess))[k+1:2k]
            if result_Z == logicalSyndromes[i,:]
                decoded += 1
            end
        end
    end
    z_error = 1 - decoded / nframes
    
    # X simulation
    fullcircuit_X = vcat(pre_X, ecirc, errors, cat, circuit_X)
    frames = PauliFrame(nframes, n+anc_qubits+k, regbits+k)
    pftrajectories(frames, fullcircuit_X)
    syndromes = pfmeasurements(frames)[:, anc_qubits+1:regbits]
    logicalSyndromes = pfmeasurements(frames)[:, regbits+1:regbits+k]

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_X = (O * stab_to_gf2(guess))[1:k]
            if result_X == logicalSyndromes[i, :]
                decoded += 1   
            end
        end
    end
    x_error = 1 - decoded / nframes
    
    return x_error, z_error
end

"""Same as [`vary_shift_errors_plot`](@ref) but uses pauli frame simulation"""
function vary_shift_errors_plot_pf(code::AbstractECC, name=string(typeof(code)))
    checks = parity_checks(code)
    vary_shift_errors_plot_pf(checks, name)
end
"""Same as [`vary_shift_errors_plot`](@ref) but uses pauli frame simulation"""
function vary_shift_errors_plot_pf(checks::Stabilizer, name="")
    title = name*" Circuit w/ Encoding Circuit PF"
    scirc, _ = naive_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(checks)

    error_rates = 0.000:0.00150:0.12
    # Uncompiled errors 
    post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, 0) for p in error_rates]
    post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, p/10) for p in error_rates]
    post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, scirc, p, p) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s10 = [post_ec_error_rates_s10[i][1] for i in eachindex(post_ec_error_rates_s10)]
    z_error_s10 = [post_ec_error_rates_s10[i][2] for i in eachindex(post_ec_error_rates_s10)]
    x_error_s100 = [post_ec_error_rates_s100[i][1] for i in eachindex(post_ec_error_rates_s100)]
    z_error_s100 = [post_ec_error_rates_s100[i][2] for i in eachindex(post_ec_error_rates_s100)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, 0) for p in error_rates]
    compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, p/10) for p in error_rates]
    compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(checks, ecirc, new_circuit, p, p) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s10 = [compiled_post_ec_error_rates_s10[i][1] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_z_error_s10 = [compiled_post_ec_error_rates_s10[i][2] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_x_error_s100 = [compiled_post_ec_error_rates_s100[i][1] for i in eachindex(compiled_post_ec_error_rates_s100)]
    compiled_z_error_s100 = [compiled_post_ec_error_rates_s100[i][2] for i in eachindex(compiled_post_ec_error_rates_s100)]
    
    # Data + Anc Compiled circuit
    s, n = size(checks)
    k = n-s

    renewed_circuit, data_order = data_ancil_reindex(scirc, n+s)
    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end
    renewed_ecirc = perfect_reindex(ecirc, data_order)
    
    dataQubits = n
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    renewed_checks = checks[:,parity_reindex] 
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, 0, encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p/10,  encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p,  encoding_locs=encoding_locs) for p in error_rates]
    full_compiled_x_error_s0 = [full_compiled_post_ec_error_rates_s0[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_z_error_s0 = [full_compiled_post_ec_error_rates_s0[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_x_error_s10 = [full_compiled_post_ec_error_rates_s10[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_z_error_s10 = [full_compiled_post_ec_error_rates_s10[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_x_error_s100 = [full_compiled_post_ec_error_rates_s100[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s100)]
    full_compiled_z_error_s100 = [full_compiled_post_ec_error_rates_s100[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s100)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, x_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, x_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_x_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_x_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_x_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
     scatter!(error_rates, full_compiled_x_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_x_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_x_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)
    
    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, z_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, z_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_z_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_z_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_z_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

     # Compiled Plots
    scatter!(error_rates, full_compiled_z_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_z_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_z_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)
    
    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end

"""Same as [`vary_shift_errors_plot`](@ref) but uses pauli frame simulation""" # TODO needs to be updated to match above - new encoding circuits broke this
function vary_shift_errors_plot_shor_syndrome(code::AbstractECC, name=string(typeof(code)))
    title = name*" Circuit - Shor Syndrome Circuit"
    checks = parity_checks(code)
    cat, scirc, anc_qubits, bit_indices = shor_syndrome_circuit(checks)
    ecirc = naive_encoding_circuit(code)

    constraints, data_qubits = size(checks)
    total_qubits = anc_qubits + data_qubits

    error_rates = 0.000:0.00150:0.12
    # Uncompiled errors 
    post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, 0) for p in error_rates]
    post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, p/10) for p in error_rates]
    post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, cat, scirc, p, p) for p in error_rates]
    x_error_s0 = [post_ec_error_rates_s0[i][1] for i in eachindex(post_ec_error_rates_s0)]
    z_error_s0 = [post_ec_error_rates_s0[i][2] for i in eachindex(post_ec_error_rates_s0)]
    x_error_s10 = [post_ec_error_rates_s10[i][1] for i in eachindex(post_ec_error_rates_s10)]
    z_error_s10 = [post_ec_error_rates_s10[i][2] for i in eachindex(post_ec_error_rates_s10)]
    x_error_s100 = [post_ec_error_rates_s100[i][1] for i in eachindex(post_ec_error_rates_s100)]
    z_error_s100 = [post_ec_error_rates_s100[i][2] for i in eachindex(post_ec_error_rates_s100)]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    new_cat = perfect_reindex(cat, order)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, 0) for p in error_rates]
    compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, p/10) for p in error_rates]
    compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(checks, ecirc, new_cat, new_circuit, p, p) for p in error_rates]
    compiled_x_error_s0 = [compiled_post_ec_error_rates_s0[i][1] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_z_error_s0 = [compiled_post_ec_error_rates_s0[i][2] for i in eachindex(compiled_post_ec_error_rates_s0)]
    compiled_x_error_s10 = [compiled_post_ec_error_rates_s10[i][1] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_z_error_s10 = [compiled_post_ec_error_rates_s10[i][2] for i in eachindex(compiled_post_ec_error_rates_s10)]
    compiled_x_error_s100 = [compiled_post_ec_error_rates_s100[i][1] for i in eachindex(compiled_post_ec_error_rates_s100)]
    compiled_z_error_s100 = [compiled_post_ec_error_rates_s100[i][2] for i in eachindex(compiled_post_ec_error_rates_s100)]

    # Data + Anc Compiled circuit
    renewed_circuit, data_order = data_ancil_reindex(scirc, total_qubits)
    renewed_ecirc = perfect_reindex(ecirc, data_order)
    renewed_cat = perfect_reindex(cat, data_order)

    # Data + Anc Compiled circuit
    s, n = size(checks)
    k = n-s

    encoding_locs = []
    for i in n-k+1:n
        push!(encoding_locs, data_order[i])
    end

    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:data_qubits)]
    renewed_checks = checks[:,parity_reindex] 
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, 0, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, p/10, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_shor_syndrome(renewed_checks, renewed_ecirc, renewed_cat, renewed_circuit, p, p, encoding_locs = encoding_locs) for p in error_rates]
    full_compiled_x_error_s0 = [full_compiled_post_ec_error_rates_s0[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_z_error_s0 = [full_compiled_post_ec_error_rates_s0[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s0)]
    full_compiled_x_error_s10 = [full_compiled_post_ec_error_rates_s10[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_z_error_s10 = [full_compiled_post_ec_error_rates_s10[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s10)]
    full_compiled_x_error_s100 = [full_compiled_post_ec_error_rates_s100[i][1] for i in eachindex(full_compiled_post_ec_error_rates_s100)]
    full_compiled_z_error_s100 = [full_compiled_post_ec_error_rates_s100[i][2] for i in eachindex(full_compiled_post_ec_error_rates_s100)]

    f_x = Figure(resolution=(1100,900))
    ax = f_x[1,1] = Axis(f_x, xlabel="single (qu)bit error rate",title=title*" Logical X")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, x_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, x_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, x_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_x_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_x_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_x_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, full_compiled_x_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_x_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_x_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)
    
    f_x[1,2] = Legend(f_x, ax, "Error Rates")

    f_z = Figure(resolution=(1100,900))
    ax = f_z[1,1] = Axis(f_z, xlabel="single (qu)bit error rate",title=title*" Logical Z")
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, z_error_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, z_error_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, z_error_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_z_error_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_z_error_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_z_error_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, full_compiled_z_error_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
    scatter!(error_rates, full_compiled_z_error_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
    scatter!(error_rates, full_compiled_z_error_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)
    
    f_z[1,2] = Legend(f_z, ax, "Error Rates")
    return f_x, f_z
end