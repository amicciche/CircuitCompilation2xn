# CircuitCompilation2xn
CircuitCompilation2xn offers some tools to manipulate syndrome extracting circuits for better use on 2xn quantum dot hardwares. For more information please refer to the [paper](https://arxiv.org/abs/2501.09061) , and if one would desire to cite this tool, please cite the paper.
## How to use this?
First, we state that this work is closely integrated with [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl) 

To begin, we will need either a parity check matrix (or stabilizer tableau), or a `Vector{QuantumClifford.AbstractOperation}` containing the syndrome extracting circuit. Currently, we enforce the constraint that syndrome extracting circuit will always be of a form where the two qubit gates act one one qubit in the data qubit region and one in the ancilla region (i.e. there are two qubit gates that go between data or ancilla qubits). 

`QuantumClifford.jl` as of present (v0.10.0), contains two functions for generating syndrome extracting circuits for any provided stabilizer tableau, one for naive syndrome extraction (a single ancillary qubit is used per parity check), and one for Shor-style fault tolerant syndrome extraction (each parity check requires a w-body GHZ state, where w is weight of the parity check at hand). This is shown in Figure 3, of the [paper](https://arxiv.org/abs/2501.09061):

![Figure 3 from paper:](assets/images/syndrome_circuits.png)

### Naive syndrome circuits example
First we will need a error correction code, and while any will do, let's use one that's already defined within `QuantumClifford.jl`, and is quite pedagogical. Furthermore, one might want to consider the X and Z checks seperately to guarantee commutativity, however for this first example and for simplicity, we will consider them together. (This is addressed later in this README)

```
using QuantumClifford
using QuantumClifford.ECC

code = Steane7()
scirc, _ = QuantumClifford.ECC.naive_syndrome_circuit(code)
```

`scirc` will now be in the style of the required input to the functions of `CircuitCompilation2xn`:
```
julia> scirc
30-element Vector{QuantumClifford.AbstractOperation}:
 sXCX(4,8)
 sXCX(5,8)
 sXCX(6,8)
 sXCX(7,8)
 sMRZ(8, 1)
 sXCX(2,9)
 sXCX(3,9)
 sXCX(6,9)
 sXCX(7,9)
 sMRZ(9, 2)
 sXCX(1,10)
 sXCX(3,10)
 sXCX(5,10)
 sXCX(7,10)
 sMRZ(10, 3)
 sCNOT(4,11)
 sCNOT(5,11)
 sCNOT(6,11)
 sCNOT(7,11)
 sMRZ(11, 4)
 sCNOT(2,12)
 sCNOT(3,12)
 sCNOT(6,12)
 sCNOT(7,12)
 sMRZ(12, 5)
 sCNOT(1,13)
 sCNOT(3,13)
 sCNOT(5,13)
 sCNOT(7,13)
 sMRZ(13, 6)
```

Visualizing this with [Quantikz.jl](https://github.com/QuantumSavory/Quantikz.jl):
![Uncompiled naive Steane7 circuit](assets/images/naive_steane_uncompiled.png)

Now we can use CircuitCompilation2xn to compile this circuit in different ways, as well as to calculate the number of shifts it would take to run in its current form.

```
using CircuitCompilation2xn
```

README under construction....