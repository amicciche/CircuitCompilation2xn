# TODO ASSUMPTION all two circuit gates will be written such that gate.q1 < gate.q2
# in other words, all q2 are ancillary qubits and all q1 are data qubits.

module CircuitCompilation2xn
using QuantumClifford
using CairoMakie
using QuantumClifford.ECC: Steane7, Shor9, naive_syndrome_circuit, shor_syndrome_circuit, naive_encoding_circuit, parity_checks, code_s, code_n, AbstractECC, faults_matrix, CSS
using Statistics
using Distributions
using NPZ
using LDPCDecoders
using SparseArrays

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

function copy(q::qblock)
    return qblock(q.elements, q.ancil)
end

# TODO this function seems fragile - could be written more robustly.
"""This function takes a circuit and returns a vector of [`qblock`](@ref) objects. These objects have values that useful for heuristics.
 This also sorts the circuit by ancillary qubit index."""
function create_blocks(circuit)
    circuit = sort!(circuit, by = x -> x.q2)
    numDataBits = circuit[1].q2 - 1 # This seems very fragile
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

# This function is really just for testing purposes - I think only for shor syndrome?
function staircase(blockset)
    i = 1
    for block in blockset
        println(block.elements[1]+i)
        i += 1
    end
end

function identity_sort(blockset)
    sorted = []
    indices = []
    sort!(blockset, by = x -> (x.elements[1]), rev=true)
    current_value = -1
    current_index = 1
    while length(indices) < length(blockset) # TODO very inefficient implement -> improve this
        if blockset[current_index].elements[1] != current_value && current_index ∉ indices
            push!(sorted, copy(blockset[current_index]))
            push!(indices, current_index)
            current_value = blockset[current_index].elements[1] 
            current_index += 1
        else
            current_index += 1
        end
        if current_index > length(blockset)
            current_value = -1
            current_index = 1
        end
    end

    order = Dict()
    sort!(blockset, by = x -> (x.ancil), rev=false)
    currentAncil = blockset[1].ancil
    for block in sorted
        order[block.ancil] = currentAncil
        currentAncil += 1
    end
    return order
end

"""Splits the provided circuit into two pieces. The first piece is the one on which we reindex. The second piece contains operations that would
cause an error in the reordering."""
function clifford_grouper(circuit)
    non_mz = Vector{QuantumClifford.AbstractOperation}()
    mz = Vector{QuantumClifford.AbstractOperation}()
    for gate in circuit
        if isa(gate, QuantumClifford.AbstractTwoQubitOperator)
            push!(non_mz, gate)
        else
            push!(mz, gate)
        end
    end
    return non_mz, mz
end

"""Inverts the data and ancil qubits. Use this function a second time after reindexing to reindex the data qubits."""
function inverter(circ, total_qubits)
    function new_index(index::Int)
        return total_qubits + 1 - index 
    end

    new_circ = Vector{QuantumClifford.AbstractOperation}()
    for gate in circ
        gate_type = typeof(gate)
        if isa(gate, QuantumClifford.AbstractTwoQubitOperator)
            push!(new_circ, gate_type(new_index(gate.q2), new_index(gate.q1)))
        elseif fieldnames(gate_type)[1] == :qubit # This should mean that the gate is a measurement
            push!(new_circ, gate_type(new_index(gate.qubit), gate.bit))
        elseif length(fieldnames(gate_type))==1 # This should mean its a single qubit gate like sX or sHadamard
            push!(new_circ, gate_type(new_index(gate.q)))
        elseif gate_type == QuantumClifford.ClassicalXOR
            push!(new_circ, gate)
        else
            println("WARNING TRIED TO INVERT SOMETHING ILL DEFINED:", gate_type)
            push!(new_circ,gate)
        end
    end
    return new_circ
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
    h1_circuit = perfect_reindex(circuit, h1_order)
    h1_batches = gate_Shuffle(h1_circuit)

    h2_order = ancil_sort_h2(blocks)
    h2_circuit = perfect_reindex(circuit, h2_order)
    h2_batches = gate_Shuffle(h2_circuit)

    h3_order = ancil_sort_h3(blocks)
    h3_circuit = perfect_reindex(circuit, h3_order)
    h3_batches = gate_Shuffle(h3_circuit)

    iden_order = identity_sort(blocks)
    iden_circuit = perfect_reindex(circuit, iden_order)
    iden_batches = gate_Shuffle(iden_circuit)

    # Returns the best reordered circuit
    if length(h1_batches)<=length(h2_batches) && length(h1_batches)<=length(iden_batches) && length(h1_batches)<=length(h3_batches)
        new_circuit = h1_circuit
        if !inverted
            new_mz = perfect_reindex(measurement_circuit,h1_order)
        end
        order = h1_order
        println("H1")
    elseif length(h2_batches)<=length(h1_batches) && length(h2_batches)<=length(iden_batches) && length(h2_batches)<=length(h3_batches)
        new_circuit = h2_circuit 
        if !inverted
            new_mz = perfect_reindex(measurement_circuit,h2_order)
        end
        order = h2_order
        println("H2")
    elseif length(h3_batches)<=length(h1_batches) && length(h3_batches)<=length(iden_batches) && length(h3_batches)<=length(h2_batches)
        new_circuit = h3_circuit
        if !inverted
            new_mz = perfect_reindex(measurement_circuit,h3_order)
        end
        order = h3_order
        println("H3")
    else 
        new_circuit = iden_circuit
        if !inverted
            new_mz = perfect_reindex(measurement_circuit,iden_order)
        end
        order = iden_order
        println("IDEN")
    end

    if inverted
        new_mz = measurement_circuit
    end
    return vcat(gate_Shuffle!(new_circuit), new_mz), order
end

"""Delta of a gate is the difference in index of the target and control bit"""
function get_delta(gate)
    if !isa(gate, QuantumClifford.AbstractTwoQubitOperator)
        return 0
    else
        return abs(gate.q2-gate.q1)
    end
end

"""Length of the returnable is the number of shifts. Splits a circuit into subcircuits that must be run in sequence due to the 2xn hardware constraints."""
function calculate_shifts(circuit)
    parallelBatches = Vector{QuantumClifford.AbstractOperation}[]
    currentDelta = -1
    for gate in circuit
        delta = get_delta(gate)
        if delta == 0
            continue
        end
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

"""Sorts by [`get_delta`](@ref) and then returns the circuirt"""
function gate_Shuffle!(circuit)
    circuit = sort!(circuit, by = x -> get_delta(x))
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

"""Sorts first by total length of the block visualation and secondarily by the ghost length"""
function ancil_sort_h3(blockset, startAncil=nothing)
    if isnothing(startAncil)
        sort!(blockset, by = x -> (x.ancil), rev=false)
        currentAncil = blockset[1].ancil
    else
        currentAncil = startAncil
    end

    sort!(blockset, by = x -> (sqrt(x.length^2 + x.ghostlength^2)), rev=true)
    order = Dict()
    
    for block in blockset
        order[block.ancil] = currentAncil
        currentAncil += 1
    end
    return order
end

"""This function should replace all other reindexing functions, and should work on an entire circuit."""
function perfect_reindex(circ, order::Dict)
    function new_index(index::Int)
       get(order,index,index) 
    end

    new_circ = Vector{QuantumClifford.AbstractOperation}()
    for gate in circ
        gate_type = typeof(gate)
        if isa(gate, QuantumClifford.AbstractTwoQubitOperator)
            push!(new_circ, gate_type(new_index(gate.q1), new_index(gate.q2)))
        elseif fieldnames(gate_type)[1] == :qubit # This should mean that the gate is a measurement
            push!(new_circ, gate_type(new_index(gate.qubit), gate.bit))
        elseif length(fieldnames(gate_type))==1 # This should mean its a single qubit gate like sX or sHadamard
            push!(new_circ, gate_type(new_index(gate.q)))
        elseif gate_type == QuantumClifford.ClassicalXOR
            push!(new_circ, gate)
        else
            println("WARNING TRIED TO REINDEX SOMETHING ILL DEFINED:", gate_type)
            push!(new_circ,gate)
        end
    end
    return new_circ
end

"""[`data_ancil_reindex`](@ref)"""
function data_ancil_reindex(code::AbstractECC)
    total_qubits = code_s(code)+code_n(code)
    scirc, _ = naive_syndrome_circuit(code)
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

    return data_reindex, merge(ancil_order, data_order) 
end

"""Returns a vector which corresponds to what the number of shifts are at different stages of compilation.
- First returns uncompiled shifts
- Second returns shifts after gate shuffles
- Third returns shifts after anc reindexing
- Fourth returns shifts after data reindexing.
""" 
function comp_numbers(circuit, total_qubits)
    shifts = []
    a, _ = clifford_grouper(circuit) # only need the two qubit gates
    push!(shifts, length(calculate_shifts(a)))
    push!(shifts, length(gate_Shuffle(a)))

    a_anc, _ = ancil_reindex_pipeline(a)
    push!(shifts, length(calculate_shifts(a_anc))) 

    a_data, = data_ancil_reindex(a, total_qubits)
    push!(shifts, length(calculate_shifts(a_data)))
    
    return shifts
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

"""Taken from the QEC Seminar notebook for plotting logical vs physical error"""
function plot_code_performance_log_log(error_rates, post_ec_error_rates; title="")
    error_rates = log10.(error_rates)
    post_ec_error_rates = log10.(post_ec_error_rates)

    f = Figure(resolution=(500,300))
    ax = f[1,1] = Axis(f, xlabel="Log10 of single (qu)bit error rate", ylabel="Log10 of logical error rate", title=title)
    #lim = max(error_rates[end],post_ec_error_rates[end])
    
    lines!([-5,0], [-5,0], label="single bit", color=:black)
    plot!(error_rates, post_ec_error_rates, label="after decoding", color=:black)
    xlims!(-5,0)
    ylims!(-5,0)
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

include("./nonpf_evalutation.jl")
include("./pf_evaluation.jl")
include("./LDPC_functions.jl")
end # module CircuitCompilation2xns