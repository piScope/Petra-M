from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


def split_AhXB_complex_mode1(Ah, X, B):
    # complex operator
    #
    #  BlockMatrix elements are arranges as
    #  [real1, real2, real3..., imag1, imag2, imag3]

    if use_parallel:
        to_opr = mfem.Opr2BlockOpr
        to_matrix = mfem.Opr2HypreParMatrix

    else:
        to_opr = mfem.Opr2BlockMatrix
        to_matrix = mfem.Opr2SparseMatrix

    Ahc = Ah.AsComplexOperator()
    BlockA_r = to_opr(Ahc.real())
    BlockA_i = to_opr(Ahc.imag())

    num_blocks = BlockA_r.NumRowBlocks()

    # this is to debug matrix
    matr = []
    mati = []
    size = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            blkr = to_matrix(BlockA_r.GetBlock(i, j))
            blki = to_matrix(BlockA_i.GetBlock(i, j))
            matr.append(blkr)
            mati.append(blki)
            if i == 0:
                size.append(blkr.Width())

    tdof_offsets = mfem.intArray(size+size)
    tdof_offsets.PartialSum()
    B.Update(tdof_offsets)
    X.Update(tdof_offsets)

    br = []
    bi = []
    for i in range(num_blocks):
        br.append(B.GetBlock(i))
        bi.append(B.GetBlock(i+num_blocks))

    xr = []
    xi = []
    for i in range(num_blocks):
        xr.append(X.GetBlock(i))
        xi.append(X.GetBlock(i+num_blocks))

    return (matr, mati), (br, bi), (xr, xi)
