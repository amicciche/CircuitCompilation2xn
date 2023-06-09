module CircuitCompilation2xn
using QuantumClifford

threeRepCode = [sCNOT(1,4),sCNOT(2,4),sCNOT(2,5),sCNOT(3,5)]
k4_example = [sCNOT(1,4),sCNOT(3,4),sCNOT(2,5),sCNOT(2,4),sCNOT(2,7),sCNOT(3,6),sCNOT(3,5)]
example_that_broke_h1= [sCNOT(5,6), sCNOT(3,6),sCNOT(1,6),sCNOT(4,7),sCNOT(2,7),sCNOT(4,8)]

struct qblock
    elements::Vector{Int}
    ancil::Int
    length::Int
    ghostlength::Int
end

function qblock(elements, ancil)
    block = qblock(sort!(elements), ancil, maximum(elements)+1, minimum(elements))
    return block
end

function print_batches(circuit_batches)
    i = 1
    for batch in circuit_batches
        println("Shift: ", i)
        for gate in batch
            println(typeof(gate)," from ", gate.q1, " to ", gate.q2)
        end
        i += 1
    end
end

# takes a circuit and strips out the measurement gates
# TODO make this less wonky - don't simply delete the measurement gates
function clifford_grouper(circuit)
    groups = []
    for i in eachindex(circuit)
        try 
            circuit[i].q1
            circuit[i].q2
            push!(groups, circuit[i])
        catch
            println("index", i, "is a measurement")
        end
    end
    return groups
end

function test(circuit=example_that_broke_h1)
    # Removes measurement gates
    circuit = clifford_grouper(circuit)

    println("Caclulate shifts without any reordering")
    print_batches(calculate_shifts(circuit))

    println("\nCaclulate shifts on same code, after delta sorting the gates")
    print_batches(gate_Shuffle(circuit))

    println("\nForm the block representation of the circuit")
    blocks = create_blocks(circuit)
    for block in blocks
        println(block)
    end

    h1_order = ancil_sort_h1(blocks)
    println("\nOrder after running heuristic 1\n", h1_order)

    println("\nShifts on delta sorted reordered h1 circuit")
    h1_batches = gate_Shuffle(ancil_reindex(circuit,h1_order))
    print_batches(h1_batches)

    h2_order = ancil_sort_h2(blocks)
    println("\nOrder after running heuristic 2\n", h2_order)

    println("\nShifts on delta sorted reordered h2 circuit")
    h2_batches = gate_Shuffle(ancil_reindex(circuit,h2_order))
    print_batches(h2_batches)

    # Returns the best reordered circuit
    if length(h1_batches)<length(h2_batches)
        return ancil_reindex(circuit, h1_order)
    else 
        return ancil_reindex(circuit, h2_order)
    end
end

# Length of the returnable is the number of shifts
function get_delta(gate)
    return gate.q2-gate.q1
end
    
function calculate_shifts(circuit)
    parallelBatches = []
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

function gate_Shuffle(circuit)
    circuit = sort(circuit, by = x -> get_delta(x))
    calculate_shifts(circuit)
end

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

function ancil_sort_h1(blockset)
    sort!(blockset, by = x -> (x.ghostlength, x.length), rev=true)
    order = []
    for block in blockset
        push!(order, block.ancil)
    end
    return order
end

function ancil_sort_h2(blockset)
    sort!(blockset, by = x -> ( x.length,x.ghostlength), rev=true)
    order = []
    for block in blockset
        push!(order, block.ancil)
    end
    return order
end

# This was written with only CNOTS in mind. It's possible this function is mixing up the gates
function ancil_reindex(circuit, order)
    new_circuit = []
    numDataBits = circuit[1].q2 - 1 # This assumes the circuit is sorted by target bits
    for gate in circuit
        gate_type = typeof(gate)
        push!(new_circuit, gate_type(gate.q1, indexin(gate.q2, order)[1]+numDataBits))
    end
    return new_circuit
end

end # module CircuitCompilation2xn