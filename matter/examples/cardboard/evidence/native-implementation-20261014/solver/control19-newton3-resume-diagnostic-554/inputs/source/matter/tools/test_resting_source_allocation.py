"""Exercise the native source-allocation guard with retired ancestry slots."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SourceAllocationTest(unittest.TestCase):
    def test_exact_source_topology_and_unused_slots(self):
        root = Path(__file__).resolve().parents[2]
        compiler = shutil.which("clang++") or shutil.which("g++")
        self.assertIsNotNone(compiler, "a C++ compiler is required")
        source = r'''#include "NumiHumanRestingSourceAllocation.hpp"
#include <cassert>
#include <limits>
int main() {
    // Retired leading/interior/trailing slots retain their source IDs.
    const std::vector<std::uint32_t> src{999,101,103,104,104,103,106,999};
    const std::vector<std::uint32_t> mesh{999,21,23,24,24,23,26,999};
    assert(numiHumanRestingSourceAllocationBase(src,1,100,8,mesh,1,6,28)==20);
    auto rejected=[&](const auto& a,std::size_t ai,std::uint32_t av,std::uint32_t nv,
                      const auto& b,std::size_t bi,std::size_t ni,std::size_t bv) {
        bool failed=false;
        try{numiHumanRestingSourceAllocationBase(a,ai,av,nv,b,bi,ni,bv);}
        catch(const std::runtime_error&){failed=true;}
        assert(failed);
    };
    auto permuted=mesh;std::swap(permuted[2],permuted[3]);
    rejected(src,1,100,8,permuted,1,6,28);
    rejected(src,1,100,8,mesh,1,6,27); // unused trailing slot exceeds pack
    rejected(src,1,100,6,mesh,1,6,28); // source index escapes allocation
    rejected(src,1,100,8,mesh,1,7,28); // incomplete triangle
    rejected(src,7,100,8,mesh,1,6,28); // index range exceeds source
    auto underflow=mesh;underflow[1]=0;
    rejected(src,1,100,8,underflow,1,6,28);
    auto out=mesh;out[3]=std::numeric_limits<std::uint32_t>::max();
    rejected(src,1,100,8,out,1,6,28);
    const std::vector<std::uint32_t> dense{0,1,2};
    assert(numiHumanRestingSourceAllocationBase(dense,0,0,3,dense,0,3,3)==0);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "allocation.cpp"
            binary = Path(directory) / "allocation"
            cpp.write_text(source)
            subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(root / "apps"), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
