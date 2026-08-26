import { cropGrid, decimateGrid, LocalresGrid } from './live';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';

function grid(nx: number, ny: number, nz: number, fill: (i:number,j:number,k:number)=>number): LocalresGrid {
    const values = new Float32Array(nx*ny*nz);
    for (let i=0;i<nx;i++) for (let j=0;j<ny;j++) for (let k=0;k<nz;k++)
        values[(i*ny+j)*nz+k] = fill(i,j,k);
    return { nx, ny, nz, origin: Vec3.create(10,20,30), stepX: Vec3.create(2,0,0),
             stepY: Vec3.create(0,2,0), stepZ: Vec3.create(0,0,2), values };
}
const at = (g: LocalresGrid, i:number,j:number,k:number) => g.values[(i*g.ny+j)*g.nz+k];
let failures = 0;
const check = (ok: boolean, what: string) => { if (!ok) { failures++; console.error("FAIL:", what); } };

// crop: a known blob at [10..14]^3 in a 32^3 grid, level 5, margin 2
{
    const g = grid(32,32,32,(i,j,k)=> (i>=10&&i<=14&&j>=10&&j<=14&&k>=10&&k<=14) ? 9 : 1);
    const c = cropGrid(g, 5);
    check(c.nx===9 && c.ny===9 && c.nz===9, `crop dims ${c.nx},${c.ny},${c.nz} != 9^3`);
    check(c.origin[0]===10+8*2 && c.origin[1]===20+8*2 && c.origin[2]===30+8*2,
          `crop origin ${c.origin} != [26,36,46]`);
    // every value must equal the source at the shifted index -- independent traversal order
    let same = true;
    for (let k=0;k<c.nz;k++) for (let i=0;i<c.nx;i++) for (let j=0;j<c.ny;j++)
        if (at(c,i,j,k) !== at(g,i+8,j+8,k+8)) same = false;
    check(same, "crop values differ from source");
    // no voxel >= level escaped the crop
    let escaped = 0;
    for (let i=0;i<g.nx;i++) for (let j=0;j<g.ny;j++) for (let k=0;k<g.nz;k++)
        if (at(g,i,j,k) >= 5 && (i<8||i>16||j<8||j>16||k<8||k>16)) escaped++;
    check(escaped===0, `${escaped} voxels >= level escaped the crop`);
}
// crop: level above the maximum -> tiny grid, nothing lost conceptually
{
    const g = grid(8,8,8,()=>1);
    const c = cropGrid(g, 99);
    check(c.nx<=4 && c.ny<=4 && c.nz<=4, `empty crop still ${c.nx},${c.ny},${c.nz}`);
}
// crop: blob touching the boundary -> clamped, values preserved
{
    const g = grid(16,12,10,(i,j,k)=> (i<=1&&j<=1&&k<=1) ? 7 : 0);
    const c = cropGrid(g, 5);
    check(c.origin[0]===10 && c.origin[1]===20 && c.origin[2]===30, "clamped crop moved the origin");
    check(at(c,0,0,0)===7, "boundary blob value lost");
}
// decimate: nearest sample at even indices, doubled steps, same origin
{
    const g = grid(9,8,7,(i,j,k)=> i*100+j*10+k);
    const d = decimateGrid(g);
    check(d.nx===5 && d.ny===4 && d.nz===4, `decimate dims ${d.nx},${d.ny},${d.nz}`);
    let same = true;
    for (let i=0;i<d.nx;i++) for (let j=0;j<d.ny;j++) for (let k=0;k<d.nz;k++) {
        const si=Math.min(2*i,8), sj=Math.min(2*j,7), sk=Math.min(2*k,6);
        if (at(d,i,j,k) !== si*100+sj*10+sk) same = false;
    }
    check(same, "decimate picked wrong samples");
    check(d.stepX[0]===4 && d.stepY[1]===4 && d.stepZ[2]===4, "decimate steps not doubled");
    check(d.origin[0]===10, "decimate moved the origin");
}
// No `process` here: this file is typechecked with the browser tsconfig. An uncaught
// throw makes node exit nonzero, which is all the harness needs.
if (failures > 0) throw new Error(`${failures} grid-helper failures`);
console.log("OK (grid helpers)");
