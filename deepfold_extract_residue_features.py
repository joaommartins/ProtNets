import sys
import os
import glob
import simtk
import simtk.openmm
import simtk.openmm.app
import simtk.unit
import Bio.PDB.Polypeptide
import numpy as np

from utils import grid


# DSSP to SS-index(H, E, C)
# H: HGI
# E: EB
# C: STC
def dssp2i(ss):
    '''Convert 8-class DSSP score into three numeric classes'''
    if ss in ("H", "G", "I"):
        return 0
    elif ss in ("E", "B"):
        return 1
    elif ss in ("S", "T", "C", " "):
        return 2
    else:
        # Used for unobserved SS
        return 3


def parse_pssm(pssm_filename):

    pssm_file = open(pssm_filename)

    items = 0
    prev_line = None
    aas = []
    aas2 = []
    sequence_profile = []
    aa_freqs = []
    sequence = ""
    for line in pssm_file:

        line = line.strip()

        if len(line) == 0:
            continue
        
        split_line = line.split()

        # Detect number of items in first iteration
        if items==0 and line[0] == "1":
            items = len(split_line)
            
            header = prev_line
            aas = header.split()[:20]
            aas2 = header.split()[20:40]

        prev_line = line
            
        if items==0 or len(split_line) != items:
            continue

        # extract residue number and name
        res_index, res_name = split_line[:2]
        sequence += res_name
        
        # remove first and last two entries
        split_line = split_line[2:-2]

        # Check that there are 40 values
        assert(len(split_line) == 40)

        # Use the first 20
        scores = split_line[:20]

        sequence_profile.append(np.zeros(20))
        for aa, score in zip(aas, scores):
            standard_index = Bio.PDB.Polypeptide.one_to_index(aa)
            sequence_profile[-1][standard_index] = score

        # Use the last 20
        freqs = split_line[20:]
        aa_freqs.append(np.zeros(20))
        for aa, freq in zip(aas2, freqs):
            standard_index = Bio.PDB.Polypeptide.one_to_index(aa)
            aa_freqs[-1][standard_index] = freq

    sequence_profile = np.array(sequence_profile)
    aa_freqs = np.array(aa_freqs)/100.

    # Apply sigmoid
    sequence_profile = 1.0/(1+np.exp(-sequence_profile))

    pssm_file.close()
    
    return sequence_profile, sequence, aa_freqs
    

def extract_aa_info(pdb_filename, pssm_filenames, dssp_executable):

    pdb_id = os.path.basename(pdb_filename).split('.')[0]

    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, pdb_filename)

    # Calculate secondary structure
    try:
        dssp = Bio.PDB.DSSP(model=structure[0], pdb_file=pdb_filename, dssp=dssp_executable)
    except:
        dssp = Bio.PDB.DSSP(model=structure[0], in_file=pdb_filename, dssp=dssp_executable)

    aa_indices_list = []
    aa_one_hot_list = []
    ss_one_hot_list = []
    pssm_list = []
    aa_freqs_list = []

    first_model = structure.get_list()[0]
    ppb = Bio.PDB.PPBuilder()
    for i, chain in enumerate(first_model):

        pps = ppb.build_peptides(chain)

        pdb_sequence = ""
        for atom in chain.get_atoms():
            if atom.id == 'CA':
                try:
                    aa = Bio.PDB.Polypeptide.three_to_one(atom.get_parent().get_resname())
                except:
                    aa = 'X'
                pdb_sequence += aa
        
        assert(len(pps) == 1 and len(pps[0])==len(pdb_sequence))
        # pdb_sequence = pps[0].get_sequence()

        ss = []
        for res in chain.get_list()[0]:
            try:
                ss.append(dssp2i(res.xtra["SS_DSSP"]))
            except:
                ss.append(3)
        ss = np.array(ss, dtype=np.int8)
        # ss = np.array([dssp2i(res.xtra["SS_DSSP"]) for res in structure[0].get_list()[0]], dtype=np.int8)
        ss_one_hot = np.zeros((len(ss), 4))
        ss_one_hot[np.arange(len(ss)), ss] = 1

        # Parse position-specif scoring matrix
        pssm, sequence, aa_freqs = parse_pssm(pssm_filenames[i])

        # parse_alignment(pdb_id, pdb_sequence, alignment_filename)

        # Add column for unobserved values
        pssm = np.hstack((pssm, np.zeros((pssm.shape[0], 1))))

        # Check that sequence in PSSM is the same as that in PDB structure
        assert(str(sequence) == str(pdb_sequence))

        # Create one-hot encoding of amino acids
        aa_indices = [Bio.PDB.Polypeptide.one_to_index(aa) for aa in sequence]
        aa_one_hot = np.zeros((len(aa_indices), 20))
        aa_one_hot[np.arange(len(aa_indices)), aa_indices] = 1

        # Accumulate in lists
        aa_indices_list.append(np.array(aa_indices))
        aa_one_hot_list.append(aa_one_hot)
        ss_one_hot_list.append(ss_one_hot)
        pssm_list.append(pssm)
        aa_freqs_list.append(aa_freqs)

    aa_indices = np.concatenate(aa_indices_list)
    aa_one_hot = np.concatenate(aa_one_hot_list)
    ss_one_hot = np.concatenate(ss_one_hot_list)
    pssm = np.concatenate(pssm_list)
    aa_freqs = np.concatenate(aa_freqs_list)

    # Add column for unobserved values
    aa_one_hot_w_unobserved = np.hstack((aa_one_hot, np.zeros((aa_one_hot.shape[0], 1))))

    # Define aa_one_hot default value
    aa_one_hot_w_unobserved_default = np.zeros(21)
    aa_one_hot_w_unobserved_default[-1] = 1 

    # Define PSSM default
    pssm_default = np.zeros(21)
    pssm_default[-1] = 1


    
    # Read in PDB file
    pdb = simtk.openmm.app.PDBFile(pdb_filename)

    # # Extract positions
    positions = pdb.getPositions()
    # CA_positions = np.array([atom.coord for atom in pps[0].get_ca_list()])

    # Create structured array for positions
    position_features = np.empty(shape=(len(positions), 1), dtype=[('name','a5'),
                                                                    ('res_index', int),
                                                                    ('x',np.float32), ('y',np.float32), ('z',np.float32)])

    # Check that number of chains is the same according to the two PDB parsers
    assert(len(list(pdb.getTopology().chains())) == len(first_model))

    # Iterate over chain,residue,atoms and extract features
    for chain in pdb.getTopology().chains():
        for residue in chain.residues():
            # print residue.id, residue.name
            for atom in residue.atoms():
                index = atom.index
                position = list(positions[index].value_in_unit(simtk.unit.angstrom))
                position_features[index] = tuple([atom.name, residue.index]+position)

    

    return pdb_id, position_features, aa_indices, aa_one_hot, aa_one_hot_w_unobserved, aa_one_hot_w_unobserved_default, pssm, pssm_default, ss, ss_one_hot, aa_freqs


def embed_in_grid(pdb_id, output_dir, features, max_radius, n_features, bins_per_angstrom, coordinate_system, z_direction):

    # Extract coordinates as normal numpy array
    position_array = features[['x','y','z']].view(np.float32)
    CA_features = features[features[['name']].view('a5') == "CA"]
    position_array_CA = CA_features[['x','y','z']].view(np.float32).reshape(CA_features.shape + (-1,))
    res_indices_CA = CA_features['res_index']
    assert(np.all(res_indices_CA == np.arange(res_indices_CA.shape[0])))
    
    res_indices = features[['res_index']].view(int)
    # atom_names =  features[['name']].view('a5')
    
    selector_list = []
    indices_list = []
    for residue_index in range(pssm.shape[0]):

        # Extract origin
        CA_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "CA"))]
        N_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "N"))]
        C_feature = features[np.argmax(np.logical_and(res_indices == residue_index, features[['name']].view('a5') == "C"))]

        pos_CA = CA_feature[['x','y','z']].view(np.float32)
        pos_N = N_feature[['x','y','z']].view(np.float32)
        pos_C = C_feature[['x','y','z']].view(np.float32)

        # Define local coordinate system
        rot_matrix = grid.define_coordinate_system(pos_N, pos_CA, pos_C, z_direction)

        # Calculate spherical coordinates
        xyz = position_array_CA - pos_CA

        # Rotate to the local reference
        xyz = np.dot(rot_matrix, xyz.T).T

        if coordinate_system == grid.CoordinateSystem.spherical:
        
            r, theta, phi = grid.cartesian_to_spherical_coordinates(xyz)

            # Create grid
            grid_matrix = grid.create_spherical_grid(max_radius=max_radius, n_features=n_features, bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            r_bin, theta_bin, phi_bin = grid.discretize_into_spherical_bins(r, theta, phi, max_radius,
                                                                            grid_matrix.shape[0],
                                                                            grid_matrix.shape[1],
                                                                            grid_matrix.shape[2])
            
            # Merge bin indices into one array
            indices = np.vstack((r_bin, theta_bin, phi_bin)).transpose()

        elif coordinate_system == grid.CoordinateSystem.cubed_sphere:

            # Convert to coordinates on the cubed sphere
            patch, r, xi, eta = grid.cartesian_to_cubed_sphere_vectorized(xyz[:, 0], xyz[:, 1], xyz[:, 2])

            # Create grid
            grid_matrix = grid.create_cubed_sphere_grid(max_radius=max_radius, n_features=n_features, bins_per_angstrom=bins_per_angstrom)

            # Bin each dimension independently
            patch_bin, r_bin, xi_bin, eta_bin = grid.discretize_into_cubed_sphere_bins(patch, r, xi, eta,
                                                                                       max_radius,
                                                                                       grid_matrix.shape[1],
                                                                                       grid_matrix.shape[2],
                                                                                       grid_matrix.shape[3])
            # Merge bin indices into one array
            indices = np.vstack((patch_bin, r_bin, xi_bin, eta_bin)).transpose()
            
            
        # Create an index array to keep track of entries within range
        # ALSO exclude features from residue itself
        selector = np.where(r < max_radius)[0]

        # Apply selector on indices array
        indices = indices[selector]
        
        # Append indices and selector for current residues to list
        indices_list.append(indices)
        selector_list.append(selector)

        # print pssm.shape
        # print pssm[selector].shape
        # print grid_matrix.shape
        # print grid_matrix[indices[:,0], indices[:,1], indices[:,2], 0:20].shape
        # print indices.shape

        # pssm_default = np.zeros(21)
        # grid_matrix[:,:,:, 0:pssm_default.shape[0]] = pssm_default
        # grid_matrix[indices[:,0], indices[:,1], indices[:,2], 0:pssm.shape[1]] = pssm[selector]
        # sys.exit()
        
    max_selector = max([len(selector) for selector in selector_list])
    selector_array = np.full((len(selector_list), max_selector), -1, dtype=np.int32)
    for i, selector in enumerate(selector_list):
        selector_array[i,:len(selector)] = selector.astype(np.int32)

    max_length = max([len(indices) for indices in indices_list])
    indices_array = np.full((len(indices_list), max_length, indices_list[0].shape[1]), -1, dtype=np.int16)
    for i, indices in enumerate(indices_list):
        indices_array[i,:len(indices)] = indices.astype(np.int16)

    # Save using numpy binary format
    np.savez_compressed(os.path.join(output_dir, "%s_residue_features"%pdb_id), indices=indices_array, selector=selector_array)
        

if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = ArgumentParser()
    parser.add_argument("--coordinate-system", choices=[e.name for e in grid.CoordinateSystem], default=grid.CoordinateSystem.spherical.name,
                        help="Which coordinate system to use (default: %(default)s)")
    parser.add_argument("--z-direction", choices=[e.name for e in grid.ZDirection], default=grid.ZDirection.outward.name,
                        help="Which direction to choose for z-axis (default: %(default)s)")
    parser.add_argument("--pdb-input-dir", dest="pdb_input_dir",
                        help="Location of pdbs")
    parser.add_argument("--seq-profile-input-dir", dest="seq_profile_input_dir",
                        help="Location of pssm files")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Where to dump features")
    parser.add_argument("--dssp-executable", dest="dssp_executable",
                        help="Location of DSSP executable")
    parser.add_argument("--max-radius", dest="max_radius", default=12.0,
                        help="Maximum radius used in grid")
    parser.add_argument("--bins-per-angstrom", dest="bins_per_angstrom", default=0.5,
                        help="How many bins to use per angstromg (in r-dimension and for maximum r)")
    parser.add_argument("--include-seq-profile", dest="include_seq_profile",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to include PSSM feature")

    options = parser.parse_args()

    coordinate_system = grid.CoordinateSystem[options.coordinate_system]
    z_direction = grid.ZDirection[options.z_direction]
    
    n_features = 21+1
    residue_features = ["aa_one_hot_w_unobserved", "residue_index"]
    if options.include_seq_profile:
        n_features += 21
        residue_features = ["aa_one_hot_w_unobserved", "pssm", "residue_index"]
    
    pdb_filenames = glob.glob(os.path.join(options.pdb_input_dir, "*.pdb"))    

    for pdb_filename in pdb_filenames:

        print pdb_filename
        
        pdb_id = os.path.basename(pdb_filename).split('.')[0]
        pssm_input_files = sorted(glob.glob(os.path.join(options.seq_profile_input_dir, pdb_id + "*.pssm")))
        # alignment_input_files = sorted(glob.glob(os.path.join(options.seq_profile_input_dir, pdb_id + "*.aln")))

        # try:
        pdb_id, position_features, aa, aa_one_hot, aa_one_hot_w_unobserved, aa_one_hot_w_unobserved_default, pssm, pssm_default, ss, ss_one_hot, aa_freqs = extract_aa_info(pdb_filename, pssm_input_files, options.dssp_executable)
        # except KeyError:        # DSSP raises a KeyError in rare cases (perhaps non-standard AA?)
        #     continue
        # except AssertionError as e:
        #     print >> sys.stderr, "AssertionError: ", e, "Skipping"

        # Save protein level features
        if not os.path.exists(options.output_dir):
            os.makedirs(options.output_dir)
        np.savez_compressed(os.path.join(options.output_dir, "%s_protein_features"%pdb_id),
                            ss = ss,
                            ss_one_hot = ss_one_hot,
                            aa_one_hot = aa_one_hot,
                            aa_one_hot_w_unobserved = aa_one_hot_w_unobserved,
                            aa_one_hot_w_unobserved_default = aa_one_hot_w_unobserved_default,
                            pssm = pssm,
                            pssm_default = pssm_default,
                            aa_freqs = aa_freqs,
                            residue_features=residue_features,
                            coordinate_system = np.array(coordinate_system.value, dtype=np.int32),
                            z_direction = np.array(z_direction.value, dtype=np.int32),
                            max_radius = np.array(options.max_radius, dtype=np.float32), # angstrom
                            n_features = np.array(n_features, dtype=np.int32),
                            bins_per_angstrom = np.array(options.bins_per_angstrom, dtype=np.float32),
                            n_residues=np.array(len(np.unique(position_features[['res_index']].view(int))), dtype=np.int32))

        # np.savez_compressed(os.path.join(output_dir, "%s_aa"%pdb_id),
        #                     aa_one_hot = aa_one_hot)
        # np.savez_compressed(os.path.join(output_dir, "%s_ss"%pdb_id),
        #                     ss=ss)
        
        embed_in_grid(pdb_id, options.output_dir, position_features, 
                      max_radius=options.max_radius,
                      n_features = n_features,
                      bins_per_angstrom = options.bins_per_angstrom,
                      coordinate_system = coordinate_system,
                      z_direction = z_direction)
        
        # sys.exit()
