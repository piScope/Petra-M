from collections import (OrderedDict,
                         defaultdict)
def edit_msh_to_add_sequential_physicals(in_file,
                                         out_file,
                                         gen_all_phys_entity=False,
                                         verbose=True):
    '''
    edit msh file and add physical entities
    works for msh2.2
    '''
    from petram.geom.read_gmsh import gmsh_element_type, gmsh_element_dim

    lines = OrderedDict()

    fid = open(in_file, 'r')

    def readline1():
        ret = fid.readline()
        new_lines1.append(ret)

    l = fid.readline()
    while l:
        line = l.strip()
        if len(line) == 0:
            l = fid.readline()
            continue

        if line[:4] == '$End':
            pass
        elif line[0] == '$':
            section = line[1:]
            lines[section] = []
        else:
            lines[section].append(line)
        l = fid.readline()
    fid.close()

    # process element
    ndims = [defaultdict(int),
             defaultdict(int),
             defaultdict(int),
             defaultdict(int)]

    elines = [[], [], [], []]

    for k, l in enumerate(lines["Elements"][1:]):
        xx = l.split(' ')
        el_type = int(xx[1])
        el_num = int(xx[4])
        xx[3] = str(el_num)
        dd = gmsh_element_dim[el_type]
        ndims[dd][el_num] = ndims[dd][el_num] + 1

        elines[dd].append(' '.join(xx))

    if not gen_all_phys_entity and len(ndims[3]) != 0:
        elines2 = elines[2] + elines[3]
        nphys = len(ndims[2]) + len(ndims[3])
        ndims[0] = {}
        ndims[1] = {}
        if verbose:
            print("Adding " + str(len(ndims[2])) + " Surface(s)")
            print("Adding " + str(len(ndims[3])) + " Volume(s)")

    elif not gen_all_phys_entity and len(ndims[2]) != 0:
        elines2 = elines[1] + elines[2]
        nphys = len(ndims[1]) + len(ndims[2])
        ndims[0] = {}
        if verbose:
            print("Adding " + str(len(ndims[1])) + " Line(s)")
            print("Adding " + str(len(ndims[2])) + " Surface(s)")

    else:
        elines2 = elines[0] + elines[1] + elines[2] + elines[3]
        nphys = len(ndims[0]) + len(ndims[1]) + \
            len(ndims[2]) + len(ndims[3])
        if verbose:
            print("Adding " + str(len(ndims[0])) + " Point(s)")
            print("Adding " + str(len(ndims[1])) + " Line(s)")
            print("Adding " + str(len(ndims[2])) + " Surfac(s)")
            print("Adding " + str(len(ndims[3])) + " Volume(s)")

    # renumber elements
    elines3 = []
    for k, l in enumerate(elines2):
        xx = l.split(' ')
        elines3.append(' '.join([str(k+1)] + xx[1:]))
    elines3 = [str(len(elines3))]+elines3
    lines["Elements"] = elines3

    phys_names = [str(nphys)]
    for l in list(ndims[0]):
        phys_names.append(" ".join(["0", str(l), '"point'+str(l)+'"']))
    for l in list(ndims[1]):
        phys_names.append(" ".join(["1", str(l), '"line'+str(l)+'"']))
    for l in list(ndims[2]):
        phys_names.append(" ".join(["2", str(l), '"surface'+str(l)+'"']))
    for l in list(ndims[3]):
        phys_names.append(" ".join(["3", str(l), '"volume'+str(l)+'"']))

    lines["PhysicalNames"] = phys_names

    def write_section(fid, lines, sec):
        fid.write("$"+sec+"\n")
        fid.write("\n".join(lines[sec])+"\n")
        fid.write("$End"+sec+"\n")

    rest_sec = [x for x in list(lines) if x !=
                "MeshFormat" and x != "PhysicalNames"]

    fid = open(out_file, "w")

    write_section(fid, lines, "MeshFormat")
    write_section(fid, lines, "PhysicalNames")
    for sec in rest_sec:
        write_section(fid, lines, sec)

    fid.close()
    #from shutil import copyfile
    #copyfile(filename, filename+'.bk')
