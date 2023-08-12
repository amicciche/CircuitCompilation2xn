# TODO ASSUMPTION all two circuit gates will be written such that gate.q1 < gate.q2
# in other words, all q2 are ancillary qubits and all q1 are data qubits.

module CircuitCompilation2xn
using QuantumClifford
using CairoMakie
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, encoding_circuit, parity_checks, code_s, code_n, AbstractECC, faults_matrix
using Statistics

threeRepCode = [sCNOT(1,4),sCNOT(2,4),sCNOT(2,5),sCNOT(3,5)]
k4_example = [sCNOT(1,4),sCNOT(3,4),sCNOT(2,5),sCNOT(2,4),sCNOT(2,7),sCNOT(3,6),sCNOT(3,5)]
example_that_broke_h1= [sCNOT(5,6), sCNOT(3,6),sCNOT(1,6),sCNOT(4,7),sCNOT(2,7),sCNOT(4,8)]

struct qblock
    elements::Vector{Int}
    ancil::Int
    length::Int # Useful for heuristics
    ghostlength::Int # Useful for heuristics
end

function qblock(elements, ancil)
    block = qblock(sort!(elements), ancil, maximum(elements)+1, minimum(elements))
    return block
end

"""Custom print for a list of circuits. This arises from [`calculate_shifts`](@ref), and [`gate_Shuffle`](@ref)"""
function print_batches(circuit_batches)
    i = 1
    for batch in circuit_batches
        #println("Shift: ", i)
        for gate in batch
            #println(typeof(gate)," from ", gate.q1, " to ", gate.q2)
        end
        i += 1
    end
    println("Total shifts:", i-1)
end

"""First returns a circuit with the measurements removed, and then returns a circuit of just measurements"""
function clifford_grouper(circuit)
    non_mz = Vector{QuantumClifford.AbstractOperation}()
    mz = Vector{QuantumClifford.AbstractOperation}()
    for i in eachindex(circuit)
        try 
            circuit[i].q1
            circuit[i].q2
            push!(non_mz, circuit[i])
        catch
            push!(mz, circuit[i])
        end
    end
    return non_mz, mz
end

"""Inverts the data and ancil qubits. Use this function a second time after reindexing to reindex the data qubits."""
function inverter(circuit, total_qubits)
    circ, measurement_circuit = clifford_grouper(circuit)
    new_circuit = Vector{QuantumClifford.AbstractOperation}();
    for gate in circ
        gate_type = typeof(gate)
        push!(new_circuit, gate_type(total_qubits + 1 - gate.q2, total_qubits + 1 - gate.q1))
    end
    for gate in measurement_circuit
        gate_type = typeof(gate)
        push!(new_circuit, gate_type(total_qubits + 1 - gate.qubit, gate.bit))
    end
    return new_circuit
end

"""Maps the ordering calculated on a circuit that was inverted [`inverter`](@ref) back to the orignal circuit."""
function invertOrder(order, total_qubits)
    invertedOrder = Dict()
    for (key,value) in order
        invertedOrder[total_qubits + 1 - key] = total_qubits + 1 - value
    end
    return invertedOrder
end

"""Runs pipline on a circuit. If using a code, ECC.naive_syndrome_circuit should be run first. Returns new circuit and ordering"""
function ancil_reindex_pipeline(circuit, inverted=false)
    circuit, measurement_circuit = clifford_grouper(circuit)
    blocks = create_blocks(circuit)

    h1_order = ancil_sort_h1(blocks)
    h1_circuit = ancil_reindex(circuit, h1_order)
    h1_batches = gate_Shuffle(h1_circuit)

    h2_order = ancil_sort_h2(blocks)
    h2_circuit = ancil_reindex(circuit, h2_order)
    h2_batches = gate_Shuffle(h2_circuit)

    # Returns the best reordered circuit
    if length(h1_batches)<length(h2_batches)
        new_circuit = h1_circuit
        if !inverted
            new_mz = ancil_reindex_mz(measurement_circuit,h1_order)
        end
        order = h1_order
    else 
        new_circuit = h2_circuit
        if !inverted
            new_mz = ancil_reindex_mz(measurement_circuit,h2_order)
        end
        order = h2_order
    end

    if inverted
        new_mz = measurement_circuit
    end
    return vcat(new_circuit, new_mz), order
end

"""Delta of a gate is the difference in index of the target and control bit"""
function get_delta(gate)
    return abs(gate.q2-gate.q1)
end

"""Length of the returnable is the number of shifts. Splits a circuit into subcircuits that must be run in sequence due to the 2xn hardware constraints.
This function can't take measurement gates,and maybe only takes two qubit gates."""
# TODO change get_delta to return a delta of 0 or some error handling to account for the current limitation
function calculate_shifts(circuit)
    parallelBatches = Vector{AbstractTwoQubitOperator}[]
    currentDelta = -1
    for gate in circuit
        delta = get_delta(gate)
        if delta != currentDelta
            currentDelta = delta
            push!(parallelBatches, [])
        end
        push!(last(parallelBatches), gate)
    end
    return parallelBatches
end

"""Sorts by [`get_delta`](@ref) and then returns list of parallel batches via [`calculate_shifts`](@ref)"""
function gate_Shuffle(circuit)
    circuit = sort(circuit, by = x -> get_delta(x))
    calculate_shifts(circuit)
end

# TODO this function seems fragile - could be written more robustly.
"""This function takes a circuit and returns a vector of [`qblock`](@ref) objects. These objects have values that useful for heuristics. This also sorts the circuit by ancillary qubit index."""
function create_blocks(circuit)
    circuit = sort!(circuit, by = x -> x.q2)
    numDataBits = circuit[1].q2 - 1
    sets = []
    currentAncil = -1
    for gate in circuit
        if currentAncil != gate.q2
            currentAncil = gate.q2
            push!(sets, (Int[], currentAncil))
        end
        push!(last(sets)[1], get_delta(gate) - (currentAncil - numDataBits))
    end
   
    blockset = []
    for set in sets
        push!(blockset, qblock(set[1], set[2]))
    end
    return blockset    
end

"""Sorts first by ghost length of the block visualation and secondarily by the total length"""
function ancil_sort_h1(blockset, startAncil=nothing)
    if isnothing(startAncil)
        sort!(blockset, by = x -> (x.ancil), rev=false)
        currentAncil = blockset[1].ancil
    else
        currentAncil = startAncil
    end

    sort!(blockset, by = x -> (x.ghostlength, x.length), rev=true)
    order = Dict()
    
    for block in blockset
        order[block.ancil] = currentAncil
        currentAncil += 1
    end
    return order
end

"""Sorts first by total length of the block visualation and secondarily by the ghost length"""
function ancil_sort_h2(blockset, startAncil=nothing)
    if isnothing(startAncil)
        sort!(blockset, by = x -> (x.ancil), rev=false)
        currentAncil = blockset[1].ancil
    else
        currentAncil = startAncil
    end

    sort!(blockset, by = x -> (x.length, x.ghostlength), rev=true)
    order = Dict()
    
    for block in blockset
        order[block.ancil] = currentAncil
        currentAncil += 1
    end
    return order
end

# This was written with only CNOTS in mind. It's possible (but very unlikely now) this function is mixing up the gates
"""Uses the order obtained by [`ancil_sort_h1`](@ref) or [`ancil_sort_h2`](@ref) to reindex the ancillary qubits in the provided circuit"""
function ancil_reindex(circuit, order::Dict)
    new_circuit = Vector{QuantumClifford.AbstractOperation}();
    for gate in circuit
        gate_type = typeof(gate)
        push!(new_circuit, gate_type(gate.q1, order[gate.q2]))
    end
    sort!(new_circuit, by = x -> get_delta(x))
    return new_circuit
end

"""Reindexes a circuit of just measurements, based on an order obtained by [`ancil_sort_h1`](@ref) or [`ancil_sort_h2`](@ref)"""
function ancil_reindex_mz(mz_circuit, order::Dict)
    new_circuit = Vector{QuantumClifford.AbstractOperation}();
    for gate in mz_circuit
        gate_type = typeof(gate)
        push!(new_circuit, gate_type(order[gate.qubit], gate.bit))
    end
    return new_circuit
end

# This function currently MIGHT have limited functionality on the gates it can handle.
"""Takes an encoding circuit, and reinexes based on the order"""
function encoding_reindex(ecirc, data_order::Dict)
    new_ecirc = Vector{QuantumClifford.AbstractOperation}();
    for gate in ecirc
        gate_type = typeof(gate)
        if length(fieldnames(gate_type))==1
            push!(new_ecirc, gate_type(data_order[gate.q]))
        else
            push!(new_ecirc, gate_type(data_order[gate.q1], data_order[gate.q2]))
        end
    end
    return new_ecirc
end

"""[`data_ancil_reindex`](@ref)"""
function data_ancil_reindex(code::AbstractECC)
    total_qubits = code_s(code)+code_n(code)
    scirc = naive_syndrome_circuit(code)
    return data_ancil_reindex(scirc, total_qubits)
end

"""Performs data and ancil reindexing based on a code. Returns both the new circuit, and the new order"""
function data_ancil_reindex(scirc, total_qubits)
    # First compile the ancil qubits
    newcirc, ancil_order = ancil_reindex_pipeline(scirc)

    # Swap ancil and data qubits
    inverted_new = inverter(newcirc, total_qubits)

    # Reindex again
    new_inverted_new, data_order = ancil_reindex_pipeline(inverted_new, true)

    # Swap the data and ancil qubits again
    data_reindex = inverter(new_inverted_new, total_qubits)

    # Invert the order to match the swap back of the data and ancil qubits
    data_order = invertOrder(data_order, total_qubits)

    return data_reindex, data_order
end

"""Inserts all possible 1 qubit Pauli errors on two circuits data qubits, after encoding. The compares them. The returned vector is for each error, how many discrepancies were caused."""
function evaluate(oldcirc, newcirc, ecirc, dataqubits, ancqubits, regbits, new_ecirc=ecirc, order=nothing)
    samples = 50
    diff = []
    types = [sX, sY, sZ]
    for gate in types
        for qubit in 1:dataqubits
            errors = gate(qubit)
            fullcirc_old = vcat(ecirc,errors,oldcirc)

            # needed for comparing against data qubit reindexing
            if isnothing(order)
                affected_bit = qubit
            else
                affected_bit = order[qubit]
            end
            errors = gate(affected_bit)
            fullcirc_new = vcat(new_ecirc,errors,newcirc)

            result_old = []
            for i in 1:samples
                bits = zeros(Bool,regbits)
                s = one(Stabilizer, dataqubits+ancqubits)
                state = Register(s,bits)

                mctrajectory!(state, fullcirc_old)
                push!(result_old, bits)
            end
            result_old = reduce(vcat, transpose.(result_old))

            result_new = []
            for i in 1:samples
                bits = zeros(Bool,regbits)
                s = one(Stabilizer, dataqubits+ancqubits)
                state = Register(s,bits)

                mctrajectory!(state, fullcirc_new)
                push!(result_new, bits)
            end
            result_new = reduce(vcat, transpose.(result_new))

            # This only works when we assume the input won't cause a nondeterministic measurement
            # instead of sum, we should check that each column has the same distribution to account for nondeterministic mesurements 
            # For Shor and Steane this works great, but might need to change this in the future
            push!(diff, sum(result_old .⊻ result_new))
        end
    end
    return diff
end

# The below three functions are from the QEC Lec 1 Notebook (with some changes)
# - create_lookup_table()
# - evaluate_code_decoder()
# - plot_code_performance()
"""Generate a lookup table for decoding single qubit errors. Maps s⃗ → e⃗."""
function create_lookup_table(code::Stabilizer)
    lookup_table = Dict()
    constraints, qubits = size(code)
    # In the case of no errors
    lookup_table[ zeros(UInt8, constraints) ] = zero(PauliOperator, qubits)
    # In the case of single bit errors
    for bit_to_be_flipped in 1:qubits
        for error_type in [single_x, single_y, single_z]
            # Generate e⃗
            error = error_type(qubits, bit_to_be_flipped)
            # Calculate s⃗
            # (check which stabilizer rows do not commute with the Pauli error)
            syndrome = comm(error, code)
            # Store s⃗ → e⃗
            lookup_table[syndrome] = error
        end
    end
    lookup_table
end

"""For a given physical bit-flip error rate, parity check matrix, and a lookup table,
estimate logical error rate, taking into account noisy circuits."""
function evaluate_code_decoder(code::Stabilizer, circuit,p; samples=1_000)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    initial_state = Register(MixedDestabilizer(code ⊗ S"Z"), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+1 # TODO hackish and ugly, needs fixing
    no_error_state = canonicalize!(stabilizerview(traceout!(copy(initial_state),qubits+1)))

    decoded = 0 # Counts correct decodings
    for sample in 1:samples
        state = copy(initial_state)
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
function evaluate_code_decoder_w_ecirc(code::Stabilizer, ecirc, circuit,p; samples=1_000)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    code_w_anc = QuantumClifford.one(Stabilizer,constraints+qubits)

    initial_state = Register(MixedDestabilizer(code_w_anc), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+constraints # TODO hackish and ugly, needs fixing
    mctrajectory!(initial_state, ecirc)

    no_error_state = canonicalize!(stabilizerview(traceout!(copy(initial_state),qubits+constraints)))
    
    decoded = 0 # Counts correct decodings
    for sample in 1:samples
        state = copy(initial_state)
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

"""PauliFrame version of [`evaluate_code_decoder_w_ecirc`](@ref) and [`evaluate_code_decoder_w_ecirc_shifts`](@ref).
If no p_shift is provided, this runs as if it there were no shift errors."""
function evaluate_code_decoder_w_ecirc_pf(checks::Stabilizer, ecirc, scirc, p_init, p_shift=0 ; nframes=10_000)   
    lookup_table = create_lookup_table(checks)
    O = faults_matrix(checks)
    circuit_Z = copy(scirc)
    circuit_X = copy(scirc)
    pre_X = sHadamard(1) # to initialize in x basis
    
    constraints, qubits = size(checks)
    regbits = constraints # This is an assumption for now

    if p_shift != 0       
        non_mz, mz = clifford_grouper(scirc)
        non_mz = calculate_shifts(non_mz)
        circuit_X = []
        circuit_Z = []

        first_shift = true
        for subcircuit in non_mz
            # Shift!
            if !first_shift
                append!(circuit_X, [PauliError(i,p_shift) for i in 1:qubits])
                append!(circuit_Z, [PauliError(i,p_shift) for i in 1:qubits])
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
    logcirc_Z = naive_syndrome_circuit(logview_Z)

    logview_X = [ logicalxview(md);]
    logcirc_X = naive_syndrome_circuit(logview_X)
    
    # Z logic circuit
    for gate in logcirc_Z
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_Z, sMRZ(gate.qubit+regbits, gate.bit+regbits))
        else 
            push!(circuit_Z, type(gate.q1, gate.q2+regbits))
        end
    end

   # X logic circuit
    for gate in logcirc_X
        type = typeof(gate)
        if type == sMRZ
            push!(circuit_X, sMRZ(gate.qubit+regbits, gate.bit+regbits))
        else 
            push!(circuit_X, type(gate.q1, gate.q2+regbits))
        end
    end

    # Z simulation
    errors = [PauliError(i,p_init) for i in 1:qubits]
    fullcircuit_Z = vcat(ecirc, errors, circuit_Z)

    frames = PauliFrame(nframes, qubits+constraints+1, regbits+1)
    pftrajectories(frames, fullcircuit_Z)
    syndromes = pfmeasurements(frames)[:, 1:regbits]
    logicalSyndromes = pfmeasurements(frames)[:, regbits+1]

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_Z = (O * stab_to_gf2(guess))[2]
            if result_Z == logicalSyndromes[i]
                decoded += 1
            end
        end
    end
    z_error = 1 - decoded / nframes
    
    # X simulation
    fullcircuit_X = vcat(pre_X, ecirc, errors, circuit_X)
    frames = PauliFrame(nframes, qubits+constraints+1, regbits+1)
    pftrajectories(frames, fullcircuit_X)
    syndromes = pfmeasurements(frames)[:, 1:regbits]
    logicalSyndromes = pfmeasurements(frames)[:, regbits+1]

    decoded = 0
    for i in 1:nframes
        row = syndromes[i,:]
        guess = get(lookup_table,row,nothing)
        if isnothing(guess)
            continue
        else
            result_X = (O * stab_to_gf2(guess))[1]
            if result_X == logicalSyndromes[i]
                decoded += 1
            end
        end
    end
    x_error = 1 - decoded / nframes

    return x_error, z_error
end

"""This _shifts version of [`evaluate_code_decoder_w_ecirc`](@ref) applies errors not only at the beginning of the syndrome circuit but after each 2xn shifting AbstractOperation"""
function evaluate_code_decoder_w_ecirc_shifts(code::Stabilizer, ecirc, circuit, p_init, p_shift; samples=500)
    lookup_table = create_lookup_table(code)

    constraints, qubits = size(code)
    code_w_anc = QuantumClifford.one(Stabilizer,constraints+qubits)

    initial_state = Register(MixedDestabilizer(code_w_anc), zeros(Bool,constraints))
    initial_state.stab.rank = qubits+constraints # TODO hackish and ugly, needs fixing
    mctrajectory!(initial_state, ecirc)

    no_error_state = canonicalize!(stabilizerview(traceout!(copy(initial_state),qubits+constraints)))
    
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
        state = copy(initial_state)
              
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

"""Taken from the QEC Seminar notebook for plotting logical vs physical error"""
function plot_code_performance(error_rates, post_ec_error_rates; title="")
    f = Figure(resolution=(500,300))
    ax = f[1,1] = Axis(f, xlabel="single (qu)bit error rate",title=title)
    ax.aspect = DataAspect()
    lim = max(error_rates[end],post_ec_error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)
    plot!(error_rates, post_ec_error_rates, label="after decoding", color=:black)
    xlims!(0,lim)
    ylims!(0,lim)
    f[1,2] = Legend(f, ax, "Error Rates")
    f
end

"""Function for generating 3plot - orignal circuit with shift errors, compiled with shift errors, and then compared to orignal without shift errors"""
function plot_code_performance_shift(error_rates, post_ec_error_rates_unsorted, post_ec_error_rates_shifts_sorted, original; title="")
    f = Figure(resolution=(900,700))
    ax = f[1,1] = Axis(f, xlabel="single (qu)bit error rate",title=title)
    lim = max(error_rates[end])
    lines!([0,lim], [0,lim], label="single bit", color=:black)
    scatter!(error_rates, post_ec_error_rates_unsorted, label="Original circuit with shift errors", color=:red)
    scatter!(error_rates, post_ec_error_rates_shifts_sorted, label="Compiled circuit with shift errors", color=:blue)
    scatter!(error_rates, original, label="Only Initial errors", color=:black)
  
    f[1,2] = Legend(f, ax, "Error Rates")
    f
end

"""Given a code, plots the compiled version and the original, varrying the probability of the error of the shifts"""
function vary_shift_errors_plot(code::AbstractECC, name=string(typeof(code)))
    title = name*" Circuit w/ Encoding Circuit"
    scirc = naive_syndrome_circuit(code)
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
    renewed_ecirc = encoding_reindex(ecirc, data_order)
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

"""Same as [`vary_shift_errors_plot`](@ref) but uses pauli frame simulation"""
function vary_shift_errors_plot_pf(code::AbstractECC, name=string(typeof(code)))
    title = name*" Circuit w/ Encoding Circuit PF"
    scirc = naive_syndrome_circuit(code)
    ecirc = encoding_circuit(code)
    checks = parity_checks(code)

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
    renewed_circuit, data_order = data_ancil_reindex(code)
    renewed_ecirc = encoding_reindex(ecirc, data_order)
    dataQubits = size(checks)[2]
    reverse_dict = Dict(value => key for (key, value) in data_order)
    parity_reindex = [reverse_dict[i] for i in collect(1:dataQubits)]
    renewed_checks = checks[:,parity_reindex] 
    full_compiled_post_ec_error_rates_s0 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, 0) for p in error_rates]
    full_compiled_post_ec_error_rates_s10 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p/10) for p in error_rates]
    full_compiled_post_ec_error_rates_s100 = [evaluate_code_decoder_w_ecirc_pf(renewed_checks, renewed_ecirc, renewed_circuit, p, p) for p in error_rates]
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
end # module CircuitCompilation2xn