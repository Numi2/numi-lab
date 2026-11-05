"""Array-backed initialization tables for the existing signed 433 compiler."""
from collections.abc import Mapping,Sequence
import numpy as np
from numi_human_tet_face_incidence import CompactTetFaceMap,_LOCAL

class _ColumnRow(Mapping):
    __slots__=("columns","row")
    def __init__(self,columns,row):self.columns=columns;self.row=row
    def __len__(self):return len(self.columns)
    def __iter__(self):return iter(self.columns)
    def __getitem__(self,key):
        value=self.columns[key][self.row]
        return value.item() if isinstance(value,np.generic) else value
    def copy(self):return dict(self)

class ArrayRows(Sequence):
    def __init__(self,columns):
        self.columns=dict(columns);counts={len(v) for v in self.columns.values()}
        if len(counts)!=1:raise ValueError("column row domains differ")
        self.count=counts.pop()
    def __len__(self):return self.count
    def __getitem__(self,row):
        if isinstance(row,slice):return [self[i] for i in range(*row.indices(len(self)))]
        if row<0:row+=self.count
        if not 0<=row<self.count:raise IndexError(row)
        return _ColumnRow(self.columns,row)

class _OrientedUses(Sequence):
    __slots__=("table","first","last")
    def __init__(self,table,first,last):self.table=table;self.first=first;self.last=last
    def __len__(self):return self.last-self.first
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        entry=int(self.table.base.order[self.first+index]);cell,lane=divmod(entry,4)
        corners=self.table.corners[self.table.export_lane[lane]]
        tri=tuple(map(int,self.table.base.tets[cell,corners]))
        return cell,tri,int(self.table.base.materials[cell])

class CompactOrientedFaceUses(Mapping):
    """Preserve original CORNER_FACES tuples and insertion/occurrence order."""
    def __init__(self,points,tets,materials,corners):
        self.corners=np.asarray(corners,np.int64)
        if self.corners.shape!=(4,3):raise ValueError("invalid tetrahedron face corners")
        wanted=[tuple(sorted(map(int,x))) for x in self.corners]
        have=[tuple(sorted(map(int,x))) for x in _LOCAL[:,:3]]
        if sorted(wanted)!=sorted(have):raise ValueError("incomplete tetrahedron face corners")
        self.export_lane=np.array([wanted.index(x) for x in have])
        self.base=CompactTetFaceMap(dict(points=points,tetrahedra=tets,
              element_ids=np.arange(len(tets)),material_indices=materials))
        first=self.base.order[self.base.starts].astype(np.int64)
        original_ordinal=(first//4)*4+self.export_lane[first%4]
        self.iteration=np.argsort(original_ordinal,kind="stable")
    def __len__(self):return len(self.base)
    def __iter__(self):
        for i in self.iteration:yield tuple(map(int,self.base.keys[i]))
    def __getitem__(self,key):
        rows=self.base[key]
        return _OrientedUses(self,rows.first,rows.last)
    def items(self):
        for i in self.iteration:
            yield tuple(map(int,self.base.keys[i])),_OrientedUses(self,int(self.base.starts[i]),int(self.base.ends[i]))

def compact_export_source(source):
    replacements=[
      ('    tets=[dict(parent_tet_row=i,material_index=int(candidate_material_index[i]),**{"parent_node"+str(k):int(tt[i,k]) for k in range(4)}) for i in range(len(tt))]',
       '    tets=ArrayRows(dict(parent_tet_row=range(len(tt)),material_index=candidate_material_index,**{"parent_node"+str(k):tt[:,k] for k in range(4)}))'),
      ('    labels=[dict(parent_tet_row=i,original_parent_tet_row=int(cp[i]),**{key:int(candidate_cell_label_arrays[key][i]) for key in keys}) for i in range(len(tt))]',
       '    labels=ArrayRows(dict(parent_tet_row=range(len(tt)),original_parent_tet_row=cp,**{key:candidate_cell_label_arrays[key] for key in keys}))'),
      ('    weights=[dict(vertex=i,**{"w"+str(k):float(ww[i,k]) for k in range(86)}) for i in range(len(p))]',
       '    weights=ArrayRows(dict(vertex=range(len(p)),**{"w"+str(k):ww[:,k] for k in range(86)}))'),
      ('    face_uses: dict[tuple[int, int, int], list[tuple[int, tuple[int, int, int], int]]] = defaultdict(list)\n'
       '    for cell, tet in enumerate(cell_tets):\n'
       '        for corner in CORNER_FACES:\n'
       '            tri = tuple(tet[j] for j in corner)\n'
       '            face_uses[tuple(sorted(tri))].append((cell, tri, cell_material[cell]))',
       '    face_uses=CompactOrientedFaceUses(p,tt,np.asarray(cell_material),CORNER_FACES)'),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError("signed 433 compact-container source pattern changed: "+old[:100])
        source=source.replace(old,new,1)
    return source
