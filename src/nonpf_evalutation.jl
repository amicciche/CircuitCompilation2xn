"""For a given physical bit-flip error rate, parity check matrix, and a lookup table,
estimate logical error rate, taking into account noisy circuits."""
function evaluate_code_decoder(code::Stabilizer, circuit,p; samples=5_000)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    initial_state = Register(MixedDestabilizer(code ⊗ S"Z"), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+1 # TODO hackish and ugly, needs fixing
    no_error_state = canonicalize!(stabilizerview(traceout!(Base.copy(initial_state),qubits+1)))

    decoded = 0 # Counts correct decodings
    for sample in 1:samples
        state = Base.copy(initial_state)
        # Generate random error
        error = random_pauli(qubits,p/3,nophase=true)
        # Apply that error to your physical system
        apply!(state, error ⊗ P"I")
        # Run the syndrome measurement circuit
        mctrajectory!(state, circuit)
        syndrome = UInt8.(bitview(state))
        # Decode the syndrome
        guess = get(lookup_table,syndrome,nothing)
        if isnothing(guess)
            continue
        end
        # Apply the suggested correction
        apply!(state, guess ⊗ P"I")
        # Check for errors
        if no_error_state == canonicalize!(stabilizerview(traceout!(state,qubits+1)))
            decoded += 1
        end
    end
    1 - decoded / samples
end

"""Differs from [`evaluate_code_decoder`](@ref) by using an encoding circuit instead of of a parity matrix for initializing the state"""
function evaluate_code_decoder_w_ecirc(code::Stabilizer, ecirc, circuit,p; samples=5_000)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    code_w_anc = QuantumClifford.one(Stabilizer,constraints+qubits)

    initial_state = Register(MixedDestabilizer(code_w_anc), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+constraints # TODO hackish and ugly, needs fixing
    mctrajectory!(initial_state, ecirc)

    no_error_state = canonicalize!(stabilizerview(traceout!(Base.copy(initial_state),qubits+constraints)))
    
    decoded = 0 # Counts correct decodings
    for sample in 1:samples
        state = Base.copy(initial_state)
        # Generate random error
        error = random_pauli(qubits,p/3,nophase=true)
        for anc in 1:constraints
            error = error ⊗ P"I"
        end
        # Apply that error to your physical system
        apply!(state, error)
        # Run the syndrome measurement circuit
        mctrajectory!(state, circuit)
        syndrome = UInt8.(bitview(state))
        # Decode the syndrome
        guess = get(lookup_table,syndrome,nothing)
        if isnothing(guess)
            continue
        end
        # Apply the suggested correction
        for anc in 1:constraints
            guess = guess ⊗ P"I"
        end
        apply!(state, guess)
        # Check for errors
        if no_error_state[1:constraints,1:qubits] == canonicalize!(stabilizerview(traceout!(state,qubits+constraints)))[1:constraints,1:qubits]
            decoded += 1
        end
    end
    1 - decoded / samples
end

"""This _shifts version of [`evaluate_code_decoder_w_ecirc`](@ref) applies errors not only at the beginning of the syndrome circuit but after each 2xn shifting AbstractOperation"""
function evaluate_code_decoder_w_ecirc_shifts(code::Stabilizer, ecirc, circuit, p_init, p_shift; samples=500)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    code_w_anc = QuantumClifford.one(Stabilizer,constraints+qubits)

    initial_state = Register(MixedDestabilizer(code_w_anc), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+constraints # TODO hackish and ugly, needs fixing
    mctrajectory!(initial_state, ecirc)

    no_error_state = canonicalize!(stabilizerview(traceout!(Base.copy(initial_state),qubits+constraints)))
    
    function apply_error!(state, qubits, constraints, p)
        # Generate random error
        error = random_pauli(qubits,p/3,nophase=true)
        for anc in 1:constraints
            error = error ⊗ P"I"
        end
        apply!(state, error)
    end

    non_mz, mz = clifford_grouper(circuit)
    non_mz = calculate_shifts(non_mz)

    decoded = 0 # Counts correct decodings
    for sample in 1:samples
        state = Base.copy(initial_state)
              
        apply_error!(state, qubits, constraints, p_init)
        first_shift = true
        # Non Measurements and shifts
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                apply_error!(state, qubits, constraints, p_shift)
            end
            # Run parallel batch
            mctrajectory!(state, subcircuit)
            first_shift = false
        end
       
        # Run measurements - should there be another set of errors before this?
        mctrajectory!(state, mz)
        
        syndrome = UInt8.(bitview(state))
        # Decode the syndrome
        guess = get(lookup_table,syndrome,nothing)
        if isnothing(guess)
            continue
        end
        # Apply the suggested correction
        for anc in 1:constraints
            guess = guess ⊗ P"I"
        end
        apply!(state, guess)
        # Check for errors
        if no_error_state[1:constraints,1:qubits] == canonicalize!(stabilizerview(traceout!(state,qubits+constraints)))[1:constraints,1:qubits]
            decoded += 1
        end
    end
    1 - decoded / samples
end

"""Given a code, plots the compiled version and the original, varrying the probability of the error of the shifts"""
function vary_shift_errors_plot(code::AbstractECC, name=string(typeof(code)))
    title = name*" Circuit w/ Encoding Circuit"
    scirc, _ = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)
    checks = parity_checks(code)

    error_rates = 0.000:0.00150:0.12
    # Uncompiled errors 
    post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, scirc, p, 0) for p in error_rates]
    post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, scirc, p, p/10) for p in error_rates]
    post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, scirc, p, p) for p in error_rates]

    # Anc Compiled circuit
    new_circuit, order = ancil_reindex_pipeline(scirc)
    compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, new_circuit, p, 0) for p in error_rates]
    compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, new_circuit, p, p/10) for p in error_rates]
    compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_shifts(checks, ecirc, new_circuit, p, p) for p in error_rates]

    # Data + Anc Compiled circuit
    renewed_circuit, data_order = data_ancil_reindex(code)
    renewed_ecirc = perfect_reindex(ecirc, data_order)
    dataQubits = size(checks)[2]
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    renewed_checks = checks[:,parity_reindex] 
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_shifts(renewed_checks, renewed_ecirc, renewed_circuit, p, 0) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_shifts(renewed_checks, renewed_ecirc, renewed_circuit, p, p/10) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_shifts(renewed_checks, renewed_ecirc, renewed_circuit, p, p) for p in error_rates]

    f = Figure(resolution=(1100,900))
    ax = f[1,1] = Axis(f, xlabel="single (qu)bit error rate",title=title)
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)

    # Uncompiled Plots
    scatter!(error_rates, post_ec_error_rates_s0, label="Original circuit with no shift errors", color=:red, marker=:circle)
    scatter!(error_rates, post_ec_error_rates_s10, label="Original circuit with shift errors = p/10", color=:red, marker=:utriangle)
    scatter!(error_rates, post_ec_error_rates_s100, label="Original circuit with shift errors = p", color=:red, marker=:star8)

    # Compiled Plots
    scatter!(error_rates, compiled_post_ec_error_rates_s0, label="Anc compiled circuit with no shift errors", color=:blue, marker=:circle)
    scatter!(error_rates, compiled_post_ec_error_rates_s10, label="Anc compiled circuit with shift errors = p/10", color=:blue, marker=:utriangle)
    scatter!(error_rates, compiled_post_ec_error_rates_s100, label="Anc compiled circuit with shift errors = p", color=:blue, marker=:star8)

     # Compiled Plots
     scatter!(error_rates, full_compiled_post_ec_error_rates_s0, label="Data + anc compiled circuit with no shift errors", color=:green, marker=:circle)
     scatter!(error_rates, full_compiled_post_ec_error_rates_s10, label="Data + anc compiled circuit with shift errors = p/10", color=:green, marker=:utriangle)
     scatter!(error_rates, full_compiled_post_ec_error_rates_s100, label="Data + anc compiled circuit with shift errors = p", color=:green, marker=:star8)
    
    f[1,2] = Legend(f, ax, "Error Rates")
    f
end