# ----------------------------------------------------------------------
# This file is part of the 'Dribble' package for percolation simulations.
# Copyright (c) 2013-2018 Alexander Urban (aurban@atomistic.net)
# ----------------------------------------------------------------------
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Mozilla Public License, v. 2.0, for more details.

"""
Particle neighbor list.

"""

from __future__ import print_function, division, unicode_literals

import numpy as np

__author__ = "Alexander Urban"
__date__ = "2013-01-19"

FINAL = -1
EPS = 100.0*np.finfo(np.float).eps


class NeighborList(object):

    def __init__(self, coordinates, lattice_vectors=None, cartesian=False,
                 types=None,  interaction_range=None,
                 natoms_per_box=10, nboxes=None, tolerance=0.05):
        """
        coordinates       Nx3 2-dimensional array whose N rows are
                          the (initial) coordinates; fractional lattice
                          coordinates are expected, unless the option
                          'cartesian' is True, or no lattice vectors
                          are specified
        lattice_vectors   3x3 2-dimensional array whose rows are
                          the lattice vectors; if no lattice vectors
                          are specified, the smalles wrapping box will
                          be determined, and coordinates will be assumed
                          to be cartesian
        cartesian         input coordinates are cartesian; they
                          will be converted to fractional coordinates
        types             (optional) vector of length N with types
                          for each coordinate (e.g., atomic species)
        interaction_range (optional) if not present, only nearest neighbors
                          are returned byt the neighbor list
        natoms_per_box    (optional) average number of atoms per box
                          when the lattice cell is partinioned into boxes
        nboxes            (optional) tuple of length 3 with the numbers
                          of boxes per lattice direction
        tolerance         accuracy for distance comparisons
        """

        if lattice_vectors is None:
            cartesian = True
            self._pbc = False
            coordinates = np.array(coordinates)
            cmin = np.min(coordinates, 0)
            cmax = np.max(coordinates, 0)
            xyz = cmax - cmin
            self._avec = np.array([[xyz[0], 0.0, 0.0],
                                   [0.0, xyz[1], 0.0],
                                   [0.0, 0.0, xyz[2]]])
            coordinates -= cmin
        else:
            self._pbc = True
            self._avec = np.array(lattice_vectors)

        # coordinates
        if cartesian:
            self._coo = self.cart2frac(coordinates)
        else:
            self._coo = np.array(coordinates)

        self._ncoo = len(self._coo)
        self._types = types
        self._range = interaction_range
        self._tol = tolerance

        if nboxes:
            self._nboxes = tuple(nboxes)
        else:
            a = np.linalg.norm(self._avec[0])
            b = np.linalg.norm(self._avec[1])
            c = np.linalg.norm(self._avec[2])
            N = max(1, round(self._ncoo/natoms_per_box))
            d = (float(a*b*c)/float(N))**(1./3.)
            self._nboxes = (int(round(a/d)), int(round(b/d)), int(round(c/d)))

        self._nboxes_tot = np.product(self._nboxes)
        self._box = np.empty(self._ncoo, dtype=int)
        self._first = np.empty(self._nboxes_tot, dtype=int)
        self._first[:] = FINAL
        self._next = np.zeros(self._ncoo, dtype=int)

        # determine `star' of periodic lattice cells within range
        self._T_latt = (np.array(
            [[0, 0, 0]] + self.star_setup(self._avec, self._range, self._tol)))

        # determine `star' of boxes that has to be checked for neighbors
        avec_box = self._avec.copy()
        avec_box[0] /= float(self._nboxes[0])
        avec_box[1] /= float(self._nboxes[1])
        avec_box[2] /= float(self._nboxes[2])
        T_box = self.star_setup(avec_box, self._range, self._tol)
        # represent these T vectors in the grid base to reduce their
        # number to the unique ones only:
        self._T_box = []
        for T in T_box:
            bid = self._box_ID(*T)
            if (not (bid in self._T_box)) and (bid != 0):
                self._T_box.append(bid)
        self._T_box = np.array(self._T_box, dtype=int)

        self._build_neighbor_list()

    @classmethod
    def from_structure(cls, structure, **kwargs):
        """
        Factory method: initialize neighbor list for an instance of
        pymatgen.core.structure.Structure.

        Keyword arguments are passed on to the regular constructor.
        """

        nbl = cls(structure.frac_coords,
                  lattice_vectors=structure.lattice.matrix,
                  types=structure.species, **kwargs)

        return nbl

    def __str__(self):
        ostr = "\n Neighbor List\n"
        ostr += " -------------\n\n"
        if self._range:
            ostr += " interaction range          : {}\n".format(self._range)
        else:
            ostr += " interaction range          : NNs only\n"
        ostr += " boxes per lattice direction:"
        ostr += " {} {} {}\n".format(*self._nboxes)
        ostr += " total number of atoms      : {}\n".format(self.num_coords)
        ostr += " av. number of atoms per box: {}\n".format(
            float(self.num_coords)/float(self._nboxes_tot))
        return ostr

    def __repr__(self):
        return self.__str__()

    @property
    def atom_types(self):
        """
        A list of the atomic species of each site, if provided
        on initialization.
        """
        return self._types

    @property
    def coords(self):
        """
        List of all coordinates.
        """
        return self._coo

    @property
    def interaction_range(self):
        """
        Interaction range.
        """
        return self._range

    @property
    def lattice_vectors(self):
        """
        Matrix of lattice vectors (in rows).
        """
        return self._avec

    @property
    def num_coords(self):
        """
        Total number of coordinates (= len(coo)).
        """
        return self._ncoo

    @property
    def num_boxes(self):
        """
        Number of boxes per lattice direction, in which the
        lattice cell has been partitioned.
        """
        return self._nboxes

    def get_possible_neighbors(self, i):
        """
        Get a list of possible neighbors of the specified coordinate ID.

        Arguments:
         i      the index of the coordinate, i.e., i of `coords[i]'

        Returns:
          List of coordinate indices that are possible neighbors.
          No distances are computed, so not IDs in the list have
          to be within range.
        """

        bid_home = self._box[i]
        nbl = self._box_contents(bid_home)
        # remove the original atom from the list
        del nbl[nbl.index(i)]
        for T in self._T_box:
            bid = self._add_T_to_bid(bid_home, T)
            nbl += self._box_contents(bid)

        return nbl

    def get_neighbors_and_distances(self, i, r=None, dr=None):
        """
        Get a list of coordinates within the interaction range.

        Arguments:
          i     the index of the coordinate, i.e., i of `coords[i]'
          r     interaction range smaller than the global range
          dr    accuracy for distance comparisons

        Returns:
          tuple (nbl, dist, Tvecs) where `nbl' is a list of the neighbors,
          `dist' is a list of the corresponding distances, and `Tvecs'
          is a list of the corresponding translation vectors.

          NoteL `nbl' may contain redundant entries that belong to
          different translation vectors and distances.
        """

        nbl = []
        dist = []
        Tvecs = []

        if not dr:
            dr = self._tol

        if r and (r <= self._range):
            r2 = (r+dr)**2
        elif self._range:
            r2 = (self._range+dr)**2
        else:
            print("Error: no range specified.")
            print("Use: get_nearest_neighbors() instead")
            return

        coo_i = self._coo[i]
        coo_i_T = -self._T_latt + coo_i
        coo_i_T = np.dot(coo_i_T, self._avec)

        # periodic images of i
        coo_j = np.dot(self._coo[i], self._avec)
        v_ij = coo_i_T - coo_j
        d2 = np.sum(v_ij*v_ij, axis=1)
        idx = (d2 <= r2) * (d2 > EPS)  # filter out the atom itself
        if np.any(idx):
            dist += list(np.sqrt(d2[idx]))
            Tvecs += list(self._T_latt[idx])
            nbl += len(d2[idx])*[i]

        # further possible neighbors
        possible = self.get_possible_neighbors(i)
        for j in possible:
            coo_j = np.dot(self._coo[j], self._avec)
            v_ij = coo_i_T - coo_j
            d2 = np.sum(v_ij*v_ij, axis=1)
            idx = (d2 <= r2)
            if np.any(idx):
                dist += list(np.sqrt(d2[idx]))
                Tvecs += list(self._T_latt[idx])
                nbl += len(d2[idx])*[j]

        return (nbl, dist, Tvecs)

    def get_nearest_neighbors(self, i, dr=0.1):
        """
        Get a list of coordinates of the nearest neighbors of atom i.

        Arguments:
          i     the index of the coordinate, i.e., i of `coords[i]'
          dr    allowed fluctuations in the nearest neighbor distance

        Returns:
          tuple (nbl, dist, Tvecs) where `nbl' is a list of the neighbors,
          `dist' is a list of the corresponding distances, and `Tvecs'
          is a list of the corresponding translation vectors.

          NoteL `nbl' may contain redundant entries that belong to
          different translation vectors and distances.

          Also note: if filtering for nearest neighbors, the output
          quantities `nbl', `dist', and `Tvecs' will all be ndarrays.

        """

        nbl = []
        dist = []
        Tvecs = []

        coo_i = self._coo[i]
        coo_i_T = -self._T_latt + coo_i
        coo_i_T = np.dot(coo_i_T, self._avec)

        d_min_min = np.linalg.norm(np.dot((1.0, 1.0, 1.0), self._avec))

        # periodic images of i
        coo_j = np.dot(self._coo[i], self._avec)
        v_ij = coo_i_T - coo_j
        d2 = np.sum(v_ij*v_ij, axis=1)
        idx = d2 > EPS
        d_min = np.sqrt(np.min(d2[idx]))
        d_min_min = min(d_min, d_min_min)
        if (d_min <= d_min_min + dr):
            idx = (d2 <= (d_min_min+dr)**2) * (d2 > EPS)
            if np.any(idx):
                dist += list(np.sqrt(d2[idx]))
                Tvecs += list(self._T_latt[idx])
                nbl += len(d2[idx])*[i]

        # further possible neighbors
        possible = self.get_possible_neighbors(i)
        for j in possible:
            coo_j = np.dot(self._coo[j], self._avec)
            v_ij = coo_i_T - coo_j
            d2 = np.sum(v_ij*v_ij, axis=1)
            d_min = np.sqrt(np.min(d2))
            if d_min + dr < d_min_min:
                nbl = []
                dist = []
                Tvecs = []
            d_min_min = min(d_min, d_min_min)
            if d_min > d_min_min + dr:
                continue
            idx = (d2 <= (d_min_min+dr)**2)
            if np.any(idx):
                dist += list(np.sqrt(d2[idx]))
                Tvecs += list(self._T_latt[idx])
                nbl += len(d2[idx])*[j]

        return (nbl, dist, Tvecs)

    def get_neighbors_and_distances_OLD(self, i, dr=0.1):
        """
        Get a list of coordinates within the interaction range.

        Arguments:
          i     the index of the coordinate, i.e., i of `coords[i]'
          dr    allowed fluctuations in `r', if the nearest neighbors
                are searched

        Returns:
          tuple (nbl, dist, Tvecs) where `nbl' is a list of the neighbors,
          `dist' is a list of the corresponding distances, and `Tvecs'
          is a list of the corresponding translation vectors.

          NoteL `nbl' may contain redundant entries that belong to
          different translation vectors and distances.

          Also note: if filtering for nearest neighbors, the output
          quantities `nbl', `dist', and `Tvecs' will all be ndarrays.
        """

        nbl = []
        dist = []
        Tvecs = []

        possible = self.get_possible_neighbors(i)
        for j in possible:
            (d, T) = self.get_pbc_distances_and_translations(i, j)
            if len(d) > 0:
                nbl += [j for n in range(len(d))]
                dist += d
                Tvecs += T

        # filter, if only nearest neighbors are wanted
        if not self._range:
            d_min = np.min(dist)
            idx = np.where(np.array(dist) < d_min + dr)
            nbl = np.array(nbl, dtype=int)[idx]
            dist = np.array(dist)[idx]
            Tvecs = np.array(Tvecs)[idx]

        return (nbl, dist, Tvecs)

    def get_pbc_distances_and_translations(self, i, j, r=None, dr=0.1):
        """
        Get all distances between coordinate i and the periodic images
        of coordinate j within the interaction range.  Also return the
        corresponding translation vectors.

        Arguments:
          i, j    two coordinate indices
          r       interaction_range; if `None' only the shortest distance
                  will be returned
          dr      allowed fluctuations in `r', if the nearest neighbors
                  are searched

        Returns:
          tuple (dist, Tvecs), where `dist' is an unsorted list of all
          distances, and Tvecs is a list of the corresponding translation
          vectors.
        """

        dist = []
        Tvecs = []
        coo_i = self._coo[i]

        if r and self._range and (r <= self._range):
            r2 = r*r
        elif self._range:
            r2 = self._range*self._range
        else:
            r2 = None

        vec_ij = self._coo[j] - coo_i
        if not r2:
            # use values for T = (0,0,0) for comparison
            cart = np.dot(vec_ij, self._avec)
            d2 = np.sum(cart*cart)
            d2_min = d2

        # now other T vectors
        for T in self._T_latt:
            cart = np.dot(np.add(vec_ij, T), self._avec)
            d2 = np.sum(cart*cart)
            if r2:
                if (d2 - EPS < r2):
                    dist.append(np.sqrt(d2))
                    Tvecs.append(T)
            else:
                if (d2 - dr <= d2_min):
                    if (d2 + dr < d2_min):
                        d2_min = d2
                        dist = [np.sqrt(d2)]
                        Tvecs = [T]
                    else:
                        dist.append(np.sqrt(d2))
                        Tvecs.append(T)

        return (dist, Tvecs)

    def cart2frac(self, cart_coords, avec=None):
        """
        Convert Cartesian coordinates to fractional lattice coordinates.

        Arguments:
          cart_coords[i,j]  j-th component of the Cartesian coordinates of
                            the i-th particle
          avec[i,j]         j-th component of the i-th lattice vector;
                            if no lattice vectors are given, self._avec
                            will be used

        Returns:
          frac_coords  ndarray with the fractional coordinates
        """

        if avec is None:
            avec = self._avec

        bvec = np.linalg.inv(avec)
        frac_coords = np.dot(np.array(cart_coords), bvec)

        return frac_coords

    def frac2cart(self, frac_coords, avec=None):
        """
        Convert fractional lattice coordinates to Cartesian coordinates.

        Arguments:
          frac_coords[i,j]   j-th component of the Cartesian coordinates of
                             the i-th particle
          avec[i,j]          j-th component of the i-th lattice vector;
                             if no lattice vectors are given, self._avec
                             will be used

        Returns:
          cart_coords  ndarray with the Cartesian coordinates
        """

        if avec is None:
            avec = self._avec

        cart_coords = np.dot(np.array(frac_coords), avec)

        return cart_coords

    def star_setup(self, lattice_vectors, interaction_range=None, dr=0.1):
        """
        Determine all translation vectors within the interaction range
        for the lattice defined by the given lattice vectors.

        Arguments:
          lattice_vectors    2-d ndarray with the lattice vectors as rows
          interaction_range  the range of the interaction; if not
                             specified, only the nearest neighbors will
                             be considered
          dr                 accuracy for distance comparisons

        Returns:
          A list containing the translation vectors.

        This routine does not only look for the positive half-star,
        but saves all needed (signed) translation vectors.  The
        memory overhead should not be severe, since we expect only
        a small number (< 100) of T vectors.
        """

        star = []
        common = []

        # The 26 immediate neighbors of the home box always have to be
        # considered.
        for ix in range(-1, 2):
            for iy in range(-1, 2):
                for iz in range(-1, 2):
                    T = (ix, iy, iz)
                    common.append(T)
                    if (ix, iy, iz) != (0, 0, 0):
                        if not (T in star):
                            star.append(T)

        """
        In the case of nearest neighbors only, we hope that the
        home box and its 26 neighbors are sufficient.  Technically
        there is no guarantee that any other coordinate is within
        these boxes, but that should only be a problem in systems
        of very low density (for which one is usually not interested
        in the nearest neighbor only) or for a very poor choice of
        box size.
        """

        if not interaction_range:
            return star

        r2 = (interaction_range + dr)**2

        # lattice vectors of the box grid
        avec = np.copy(lattice_vectors)

        # sign permutations:
        signs = [[1, 1, 1],
                 [-1, 1, 1],
                 [1, -1, 1],
                 [1, 1, -1],
                 [1, -1, -1],
                 [-1, 1, -1],
                 [-1, -1, 1],
                 [-1, -1, -1]]

        """
        Now the following bunch of code loops over increasingly long
        translation vectors and checks, if they are still within the
        interaction range.  If so, all boxes with a corner that can be
        reached with this T vector are added to the star.
        """

        ix = 0
        while True:
            iy = 0
            found_one_x = False
            while True:
                iz = 0
                found_one_y = False
                while True:
                    if (ix, iy, iz) == (0, 0, 0):
                        iz += 1
                        continue

                    vec = np.dot((ix, iy, iz), avec)
                    d2 = np.sum(vec*vec)
                    if (d2 - EPS < r2):

                        """
                        The distances from each corner of the home box
                        to each corner of the remote box have to be
                        checked.  So if d2 is in range, we need to
                        add all boxes that share the corner
                        (ix,iy,iz) and (-ix,-iy,-iz):
                        """

                        for T in common:
                            for s in signs:
                                T_new = tuple(np.add((ix, iy, iz), T)*s)
                                if (not (T_new in star)) and (
                                        not np.all(T_new == (0, 0, 0))):
                                    star.append(T_new)

                        found_one_y = True
                    else:
                        break

                    iz += 1
                if found_one_y:
                    iy += 1
                    found_one_x = True
                else:
                    break
            if found_one_x:
                ix += 1
            else:
                break

        return star

    def _build_neighbor_list(self):
        """
        Divide cell into boxes and assign each coordinate
        """

        self._wrap_to_home_cell()

        # assign each coordinate to a box
        for i in range(self._ncoo):
            na = int(np.floor(self._coo[i][0]*self._nboxes[0]))
            nb = int(np.floor(self._coo[i][1]*self._nboxes[1]))
            nc = int(np.floor(self._coo[i][2]*self._nboxes[2]))
            bid = self._box_ID(na, nb, nc)
            self._box[i] = bid
            self._add_to_box(bid, i)

    def _wrap_to_home_cell(self):
        """
        Wrap all coordinates to [0:1[ interval.
        """

        for coo in self._coo:
            for i in range(3):
                while coo[i] < 0.0:
                    coo[i] += 1.0
                while coo[i] >= 1.0:
                    coo[i] -= 1.0

    def _box_ID(self, na, nb, nc):
        """
        Get the box ID for a particular coordinate vector COO.
        """

        Nba = self._nboxes[0]
        Nbb = self._nboxes[1]
        Nbc = self._nboxes[2]

        bid = int(((na + Nba) % Nba))
        bid += int(((nb + Nbb) % Nbb)*Nba)
        bid += int(((nc + Nbc) % Nbc)*Nbb*Nba)

        return bid

    def _box_nabc(self, bid):
        """
        Get box coordinates na, nb, nc of box BID.
        Returns tuple: (na, nb, nc)
        """

        nbox10 = self._nboxes[1]*self._nboxes[0]
        nc = int(bid/nbox10)
        rest = bid % nbox10
        nb = int(rest/self._nboxes[0])
        na = rest % self._nboxes[0]

        return (na, nb, nc)

    def _add_T_to_bid(self, bid, T):
        """
        Add integer-mapped translation vector T to the
        box ID BID.
        """

        (na, nb, nc) = self._box_nabc(bid)
        (ta, tb, tc) = self._box_nabc(T)
        return self._box_ID(*np.add((na, nb, nc), (ta, tb, tc)))

    def _add_to_box(self, bid, i):
        """
        Add coordinate I to box BID.
        """

        self._next[i] = self._first[bid]
        self._first[bid] = i

    def _del_from_box(self, bid, i):
        """
        Remove coordinate I from box BID.
        """

        j = self._first[bid]
        if j == i:
            self._first[bid] = self._next[j]
        else:
            while self._next[j] != i:
                j = self._next[j]
            self._next[j] = self._next[i]

    def _box_contents(self, bid):
        """
        Return list of all coordinates IDs of box BID.
        """

        ids = []
        i = self._first[bid]
        if i != FINAL:
            ids.append(i)
        while self._next[i] != FINAL:
            i = self._next[i]
            ids.append(i)

        return ids


if __name__ == "__main__":

    # ---------------------------- Test 1 ---------------------------- #

    print("\n Test 1: FCC primitive unit cell (a = 1.0), NNs only")
    print(" ---------------------------------------------------")

    avec = np.array([[0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]])*1.0

    coo = np.array([[0.0, 0.0, 0.0]])

    nbl = NeighborList(coo, lattice_vectors=avec)

    print(nbl)

    print(" checking number of nearest neighbors (12 for FCC) ... ", end="")
    (nn, dist, T) = nbl.get_nearest_neighbors(0)
    if (len(nn) != 12):
        print("FAILED!\n")
        print(" {} neighbors found: {}".format(len(nn), nn))
    else:
        print("passed.\n")
        print(" {} neighbors found".format(len(nn)))
        print(" distance = {}".format(dist[0]))

    print("")

    # ---------------------------- Test 2 ---------------------------- #

    print("\n Test 2: FCC primitive unit cell (a = 1.0), r_cut = 2a")
    print(" -----------------------------------------------------")

    a = 1.0

    avec = np.array([[0.0, 0.5, 0.5],
                     [0.5, 0.0, 0.5],
                     [0.5, 0.5, 0.0]])*a

    coo = np.array([[0.0, 0.0, 0.0]])

    nbl = NeighborList(coo, lattice_vectors=avec, interaction_range=2.0*a)

    print(nbl)

    print(" neighbors within d = 2a = {}\n".format(2*a))
    (nn, dist, T) = nbl.get_neighbors_and_distances(0)

    idx = np.argsort(dist)
    n = 1
    d = dist[idx[0]]
    for i in idx[1:]:
        if (abs(dist[i]-d) < 0.05):
            n += 1
        else:
            print(" {:3d} sites at distance d/a = {}".format(n, d))
            d = dist[i]
            n = 1
    print(" {:3d} sites at distance d/a = {}".format(n, d))

    # ---------------------------- Test 3 ---------------------------- #

    print("\n Test 3: same, but FCC conventional cell and "
          "Cartesian coordinates")
    print(" -----------------------------------------------------------------")

    a = 2.0

    avec = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])*a

    coo = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]])

    nbl = NeighborList(coo, lattice_vectors=avec, cartesian=True,
                       interaction_range=2.0*a)

    print(nbl)

    print(" neighbors within d = 2a = {}\n".format(2*a))
    (nn, dist, T) = nbl.get_neighbors_and_distances(0)

    idx = np.argsort(dist)
    n = 1
    d = dist[idx[0]]
    for i in idx[1:]:
        if (abs(dist[i]-d) < 0.05):
            n += 1
        else:
            print(" {:3d} sites at distance d/a = {}".format(n, d/a))
            d = dist[i]
            n = 1
    print(" {:3d} sites at distance d/a = {}".format(n, d/a))

    print("")

    # ---------------------------- Test 4 ---------------------------- #

    print("\n Test 4: rock salt structure")
    print(" -----------------------------------------------------------------")

    a = 5.64

    avec = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])*a

    coo = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.5, 0.0]])

    types = ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']

    nbl = NeighborList(coo, lattice_vectors=avec, types=types,
                       interaction_range=2.0*a)

    print(nbl)

    print(" neighbors within d = 2a = {} of a {} atom\n".format(2*a, types[0]))
    (nn, dist, T) = nbl.get_neighbors_and_distances(0)

    idx = np.argsort(dist)
    n = 1
    d = dist[idx[0]]
    for i in idx[1:]:
        if (abs(dist[i]-d) < 0.05):
            n += 1
            t = types[nn[i]]
        else:
            print(" {:3d} sites of type {} at distance d/a = {}".format(
                n, t, d/a))
            d = dist[i]
            n = 1
    print(" {:3d} sites of type {} at distance d/a = {}".format(n, t, d/a))

    print("")
