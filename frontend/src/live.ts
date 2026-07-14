/**
 * Live coordinate streaming for pxviewer.
 *
 * The whole point is a Level-1, "as in-place as Mol* allows" update: parse the
 * topology (hierarchy + bonds) exactly once, then on every frame swap only the
 * conformation. We do that with a tiny custom state transform, `LiveTrajectory`,
 * that turns a topology `Model` plus a bare xyz frame into a one-frame
 * `Trajectory` via `Model.trajectoryFromModelAndCoordinates` (which reuses the
 * topology model and replaces just `atomicConformation`). Bumping the transform's
 * params re-runs only Model -> Structure -> Representation, and the representation
 * does a coordinate-only geometry update rather than a full rebuild.
 */

import { PluginStateObject as SO, PluginStateTransform } from 'molstar/lib/mol-plugin-state/objects';
import { StateObjectSelector, StateTransformer } from 'molstar/lib/mol-state';
import { PluginContext } from 'molstar/lib/mol-plugin/context';
import { Task } from 'molstar/lib/mol-task';
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition';
import { Model, Structure, StructureElement, StructureProperties, StructureSelection, Unit } from 'molstar/lib/mol-model/structure';
import { Coordinates, Frame, Time } from 'molstar/lib/mol-model/structure/coordinates';
import { OrderedSet, SortedArray } from 'molstar/lib/mol-data/int';
import { Script } from 'molstar/lib/mol-script/script';
import { transpiler as pymolTranspiler } from 'molstar/lib/mol-script/transpilers/pymol/parser';
import type { Canvas3DProps } from 'molstar/lib/mol-canvas3d/canvas3d';
import { decodeColor } from 'molstar/lib/mol-util/color/utils';

export interface AtomInfo {
    id: number;
    name: string;
    resname: string;
    resseq: number;
    chain: string;
}

interface LiveTrajectoryParams {
    version: number;
    x: ArrayLike<number>;
    y: ArrayLike<number>;
    z: ArrayLike<number>;
}

/** Model (topology) + a single xyz frame -> one-frame Trajectory. */
export const LiveTrajectory = PluginStateTransform.BuiltIn({
    name: 'pxviewer-live-trajectory',
    display: { name: 'pxviewer Live Trajectory', description: 'One-frame trajectory driven by streamed coordinates.' },
    from: SO.Molecule.Model,
    to: SO.Molecule.Trajectory,
    params: {
        version: PD.Numeric(0),
        x: PD.Value<ArrayLike<number>>(new Float32Array(0), { isHidden: true }),
        y: PD.Value<ArrayLike<number>>(new Float32Array(0), { isHidden: true }),
        z: PD.Value<ArrayLike<number>>(new Float32Array(0), { isHidden: true }),
    },
})({
    apply({ a, params }) {
        return Task.create('pxviewer Live Trajectory', async () => {
            const model = a.data;
            const elementCount = model.atomicHierarchy.atoms._rowCount;
            const p = params as LiveTrajectoryParams;
            if (p.x.length !== elementCount) {
                throw new Error(
                    `frame has ${p.x.length} atoms but topology has ${elementCount}; ` +
                    `coordinates must be positionally aligned to the topology`
                );
            }
            const frame: Frame = {
                elementCount,
                time: Time(p.version, 'step'),
                x: p.x,
                y: p.y,
                z: p.z,
                xyzOrdering: { isIdentity: true },
            };
            const coordinates = Coordinates.create([frame], Time(1, 'step'), Time(0, 'step'));
            const trajectory = Model.trajectoryFromModelAndCoordinates(model, coordinates);
            return new SO.Molecule.Trajectory(trajectory, {
                label: 'Live Trajectory',
                description: `frame ${p.version}`,
            });
        });
    },
});
type LiveTrajectory = typeof LiveTrajectory;

function deinterleave(flat: ArrayLike<number>, n: number) {
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = flat[i * 3];
        y[i] = flat[i * 3 + 1];
        z[i] = flat[i * 3 + 2];
    }
    return { x, y, z };
}

/**
 * Builds the Mol* state tree once from a topology BinaryCIF and exposes an
 * `update()` that swaps coordinates in place. Also forwards pick events.
 */
export class LiveViewer {
    private liveTraj!: StateObjectSelector<SO.Molecule.Trajectory, LiveTrajectory>;
    private structure!: StateObjectSelector<SO.Molecule.Structure>;
    private version = 0;
    private nAtoms = 0;
    private highlightIndices: number[] = [];

    private constructor(private plugin: PluginContext) {}

    static async create(
        plugin: PluginContext,
        topologyBcif: Uint8Array,
        onPick?: (info: AtomInfo | null) => void,
    ): Promise<LiveViewer> {
        const viewer = new LiveViewer(plugin);
        await viewer.build(topologyBcif);
        if (onPick) viewer.subscribePick(onPick);
        return viewer;
    }

    private async build(topologyBcif: Uint8Array) {
        const plugin = this.plugin;
        // Copy into a fresh ArrayBuffer-backed view (rawData wants Uint8Array<ArrayBuffer>).
        const bytes = new Uint8Array(topologyBcif);
        const data = await plugin.builders.data.rawData({ data: bytes, label: 'pxviewer-topology' });
        const topologyTraj = await plugin.builders.structure.parseTrajectory(data, 'mmcif');
        const topologyModel = await plugin.builders.structure.createModel(topologyTraj);

        const model = topologyModel.obj!.data as Model;
        this.nAtoms = model.atomicHierarchy.atoms._rowCount;
        const conf = model.atomicConformation;

        // Seed the live trajectory with the topology's own coordinates so there is
        // something on screen before the first streamed frame arrives.
        const build = plugin.state.data.build().to(topologyModel).apply(LiveTrajectory, {
            version: this.version,
            x: Float32Array.from(conf.x),
            y: Float32Array.from(conf.y),
            z: Float32Array.from(conf.z),
        });
        this.liveTraj = build.selector;
        await build.commit();

        const liveModel = await plugin.builders.structure.createModel(this.liveTraj);
        const structure = await plugin.builders.structure.createStructure(liveModel);
        this.structure = structure;
        await plugin.builders.structure.representation.addRepresentation(structure, {
            type: 'ball-and-stick',
            color: 'element-symbol',
        });
    }

    /** Swap in a new frame given interleaved [x0,y0,z0,x1,...] coordinates. */
    async update(interleaved: ArrayLike<number>) {
        const { x, y, z } = deinterleave(interleaved, this.nAtoms);
        this.version += 1;
        const version = this.version;
        await this.plugin.state.data
            .build()
            .to(this.liveTraj)
            .update((old: LiveTrajectoryParams) => ({ ...old, version, x, y, z }))
            .commit();
        // A frame rebuilds the structure, so re-apply any active highlight to it.
        if (this.highlightIndices.length) this.reapplyHighlight();
    }

    /**
     * Resolve a PyMOL selection against the current structure and show it. With
     * `highlight` the matched atoms get the selection overlay; with `focus` the
     * camera zooms to them. Returns the matched positional atom indices.
     */
    applySelection(expression: string, opts: { highlight: boolean; focus: boolean }): number[] {
        const structure = this.currentStructure();
        if (!structure) return [];
        const expr = expression.trim();
        if (expr === '') {
            if (opts.highlight) this.clearSelection();
            return [];
        }
        const parsed = pymolTranspiler(expr); // throws on invalid PyMOL syntax
        const selection = Script.getStructureSelection(parsed, structure);
        const loci = StructureSelection.toLociWithSourceUnits(selection);
        const indices = collectElementIndices(loci);
        if (opts.highlight) {
            this.highlightIndices = indices;
            this.showHighlight(structure, indices);
        }
        if (opts.focus && indices.length) {
            this.plugin.managers.camera.focusLoci(loci);
        }
        return indices;
    }

    /** Clear any active highlight. */
    clearSelection() {
        this.highlightIndices = [];
        this.plugin.managers.structure.selection.clear();
    }

    private currentStructure(): Structure | undefined {
        return this.structure?.obj?.data as Structure | undefined;
    }

    private showHighlight(structure: Structure, indices: number[]) {
        const selection = this.plugin.managers.structure.selection;
        selection.clear();
        if (indices.length) {
            selection.fromLoci('set', lociFromElementIndices(structure, indices));
        }
    }

    // Indices are topology-stable, so rebuild the loci against the fresh structure.
    private reapplyHighlight() {
        const structure = this.currentStructure();
        if (structure) this.showHighlight(structure, this.highlightIndices);
    }

    private subscribePick(onPick: (info: AtomInfo | null) => void) {
        this.plugin.behaviors.interaction.click.subscribe((e) => {
            const loci = e.current.loci;
            if (!StructureElement.Loci.is(loci)) {
                onPick(null);
                return;
            }
            const location = StructureElement.Loci.getFirstLocation(loci);
            if (!location) {
                onPick(null);
                return;
            }
            onPick({
                id: StructureProperties.atom.id(location),
                name: StructureProperties.atom.label_atom_id(location),
                resname: StructureProperties.atom.label_comp_id(location),
                resseq: StructureProperties.residue.label_seq_id(location),
                chain: StructureProperties.chain.label_asym_id(location),
            });
        });
    }
}

/** Positional atom rows (model element indices) covered by an element loci. */
function collectElementIndices(loci: StructureElement.Loci): number[] {
    const set = new Set<number>();
    StructureElement.Loci.forEachLocation(loci, (loc) => { set.add(loc.element as unknown as number); });
    return Array.from(set).sort((a, b) => a - b);
}

/** Build an element loci for the given positional atom indices against a structure. */
function lociFromElementIndices(structure: Structure, indices: number[]): StructureElement.Loci {
    const want = new Set(indices);
    const elements: { unit: Unit; indices: OrderedSet }[] = [];
    for (const unit of structure.units) {
        const us = unit.elements;
        const size = OrderedSet.size(us);
        const positions: number[] = [];
        for (let i = 0; i < size; i++) {
            if (want.has(OrderedSet.getAt(us, i) as unknown as number)) positions.push(i);
        }
        if (positions.length) {
            elements.push({ unit, indices: SortedArray.ofSortedArray(positions) });
        }
    }
    return StructureElement.Loci(structure, elements as any);
}

const TAG_TOPOLOGY = 0;
const TAG_FRAME = 1;

async function setAxis(plugin: PluginContext, visible: boolean) {
    await plugin.canvas3dInitialized;
    if (!plugin.canvas3d) return;
    plugin.canvas3d.setProps((p: Canvas3DProps) => {
        p.camera.helper.axes.name = visible ? 'on' : 'off';
    });
}

async function setVolumeColor(plugin: PluginContext, ref: string, color: string) {
    const repr = await findVolumeReprCell(plugin, ref);
    if (!repr) return;
    await plugin.state.data.build().to(repr.transform.ref).update((old: any) => {
        old.colorTheme = { name: 'uniform', params: { value: decodeColor(color) } };
    }).commit();
}

async function setVolumeOpacity(plugin: PluginContext, ref: string, opacity: number) {
    const repr = await findVolumeReprCell(plugin, ref);
    if (!repr) return;
    await plugin.state.data.build().to(repr.transform.ref).update((old: any) => {
        if (old.type?.name === 'isosurface') {
            old.type.params.alpha = opacity;
        }
    }).commit();
}

const STYLE_VISUALS: Record<string, string[]> = {
    surface: ['solid'],
    wireframe: ['wireframe'],
    mesh: ['solid', 'wireframe'],
};

async function setVolumeStyle(plugin: PluginContext, ref: string, style: string) {
    const repr = await findVolumeReprCell(plugin, ref);
    if (!repr) return;
    const visuals = STYLE_VISUALS[style.toLowerCase()];
    if (!visuals) {
        console.warn('Unknown volume style:', style);
        return;
    }
    await plugin.state.data.build().to(repr.transform.ref).update((old: any) => {
        if (old.type?.name === 'isosurface') {
            old.type.params.visuals = visuals;
        }
    }).commit();
}

async function setVolumePosition(plugin: PluginContext, ref: string, position: [number, number, number]) {
    const cell = await findVolumeCell(plugin, ref);
    if (!cell) return;
    const [x, y, z] = position;
    await plugin.state.data.build().to(cell.transform.ref).update((old: any) => {
        if (old.transform?.name === 'matrix' && old.transform.params?.data) {
            const data = old.transform.params.data;
            data[12] = x;
            data[13] = y;
            data[14] = z;
        } else if (old.transform?.name === 'components' && old.transform.params?.translation) {
            old.transform.params.translation[0] = x;
            old.transform.params.translation[1] = y;
            old.transform.params.translation[2] = z;
        } else {
            console.warn('Volume', ref, 'does not have a position transform; set Volume.position to enable live position updates.');
        }
    }).commit();
}

async function findVolumeCell(plugin: PluginContext, ref: string) {
    const tag = `mvs-ref:${ref}`;
    for (let i = 0; i < 200; i++) {
        const cells = plugin.state.data.selectQ((q: any) => q.root.subtree().withTag(tag));
        if (cells.length) return cells[0];
        await new Promise((r) => setTimeout(r, 25));
    }
    return undefined;
}

async function findVolumeReprCell(plugin: PluginContext, ref: string) {
    const tag = `mvs-ref:${ref}-repr`;
    for (let i = 0; i < 200; i++) {
        const cells = plugin.state.data.selectQ((q: any) => q.root.subtree().withTag(tag));
        if (cells.length) return cells[0];
        await new Promise((r) => setTimeout(r, 25));
    }
    return undefined;
}

export interface LiveConnectionHandle {
    close(): void;
}

/**
 * Connect to a pxviewer `LiveSession` WebSocket, build the viewer from the
 * topology message, and drive it from streamed frames. Pick events are sent back.
 */
export function connectLive(plugin: PluginContext, url: string): LiveConnectionHandle {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    let viewer: LiveViewer | null = null;
    let building = false;

    ws.onopen = () => {
        console.log('pxviewer live connected to', url);
    };
    ws.onerror = (err) => {
        console.error('pxviewer live WebSocket error for', url, err);
    };
    ws.onclose = (ev) => {
        if (!ev.wasClean) {
            console.error('pxviewer live WebSocket closed unexpectedly:', ev.code, ev.reason);
        }
    };

    ws.onmessage = async (ev) => {
        if (typeof ev.data === 'string') {
            // Server -> client control messages (JSON text).
            let msg: any;
            try { msg = JSON.parse(ev.data); } catch { return; }
            if (msg.type === 'axis' && typeof msg.visible === 'boolean') {
                await setAxis(plugin, msg.visible);
            } else if (msg.type === 'volume_color' && typeof msg.ref === 'string' && typeof msg.color === 'string') {
                await setVolumeColor(plugin, msg.ref, msg.color);
            } else if (msg.type === 'volume_opacity' && typeof msg.ref === 'string' && typeof msg.opacity === 'number') {
                await setVolumeOpacity(plugin, msg.ref, msg.opacity);
            } else if (msg.type === 'volume_style' && typeof msg.ref === 'string' && typeof msg.style === 'string') {
                await setVolumeStyle(plugin, msg.ref, msg.style);
            } else if (msg.type === 'volume_position' && typeof msg.ref === 'string' && Array.isArray(msg.position) && msg.position.length === 3) {
                await setVolumePosition(plugin, msg.ref, msg.position);
            } else if (msg.type === 'select' && viewer) {
                // Resolve a PyMOL selection in the viewer and echo the matched
                // atom indices back so Python knows what was selected.
                let indices: number[] = [];
                let error: string | undefined;
                try {
                    indices = viewer.applySelection(String(msg.expression ?? ''), {
                        highlight: !!msg.highlight,
                        focus: !!msg.focus,
                    });
                } catch (e) {
                    error = e instanceof Error ? e.message : String(e);
                }
                ws.send(JSON.stringify({ type: 'selection-result', reqId: msg.reqId, indices, error }));
            }
            return;
        }
        const buffer = ev.data as ArrayBuffer;
        const tag = new DataView(buffer).getUint32(0, true);

        if (tag === TAG_TOPOLOGY) {
            if (building || viewer) return;
            building = true;
            const bcif = new Uint8Array(buffer, 4);
            viewer = await LiveViewer.create(plugin, bcif, (info) => {
                ws.send(JSON.stringify({ type: 'pick', empty: info === null, atom: info ?? undefined }));
            });
            building = false;
            ws.send(JSON.stringify({ type: 'ready' }));
        } else if (tag === TAG_FRAME && viewer) {
            // [u32 tag][u32 frameIndex][f32 * 3N]; coordinates start at byte 8.
            const coords = new Float32Array(buffer, 8);
            await viewer.update(coords);
        }
    };

    return { close: () => ws.close() };
}
