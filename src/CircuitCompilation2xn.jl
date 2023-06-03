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
            println("Gate from ", gate.q1, " to ", gate.q2)
        end
        i += 1
    end
end

function test(circuit=example_that_broke_h1)
    println("Caclulate shifts on 3 rep code")
    print_batches(calculate_shifts(circuit))

    println("\nCaclulate shifts on same code, after delta sorting the gates")
    print_batches(gate_Shuffle(circuit))

    println("\nForm the block representation of the circuit")
    blocks = create_blocks(circuit)
    for block in blocks
        println(block)
    end

    order = ancil_sort_h1(blocks)
    println("\nOrder after running heuristic 1\n", order)

    println("\nShifts on delta sorted reordered h1 circuit")
    print_batches(gate_Shuffle(ancil_reindex(circuit,order)))

    order = ancil_sort_h2(blocks)
    println("\nOrder after running heuristic 2\n", order)

    println("\nShifts on delta sorted reordered h2 circuit")
    print_batches(gate_Shuffle(ancil_reindex(circuit,order)))
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

# TODO pull the delta= intial configuration set of gates to the front
function gate_Shuffle(circuit)
    circuit = sort(circuit, by = x -> get_delta(x))
   calculate_shifts(circuit)
end


function create_blocks(circuit)
    # create sets (orig_ancil_index, primal_set)
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

function ancil_reindex(circuit, order)
    new_circuit = []
    numDataBits = circuit[1].q2 - 1 # This assumes the circuit is sorted by target bits
    for gate in circuit
        push!(new_circuit, sCNOT(gate.q1, indexin(gate.q2, order)[1]+numDataBits))
    end
    return new_circuit
end

end # module CircuitCompilation2xn