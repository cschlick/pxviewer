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
import { Model, Structure, StructureElement, Unit } from 'molstar/lib/mol-model/structure';
import { StructureProperties } from 'molstar/lib/mol-model/structure';
import { Coordinates, Frame, Time } from 'molstar/lib/mol-model/structure/coordinates';
import { OrderedSet, SortedArray } from 'molstar/lib/mol-data/int';
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
    private highlightLoci: StructureElement.Loci | undefined;
    private primitives = new Map<string, StateObjectSelector>();
    private reprNodes: StateObjectSelector[] = [];
    private clickMode = 'off';
    private mouseSelectionSet = new Set<number>();
    private measurePending: number[] = [];
    private pickHandler?: (info: AtomInfo | null) => void;
    /** Set by the connection to report a click-built selection back to Python. */
    onSelectionChange?: (indices: number[]) => void;
    /** Set by the connection to report a click-built measurement back to Python. */
    onMeasure?: (kind: string, atoms: number[]) => void;

    private constructor(private plugin: PluginContext) {}

    static async create(
        plugin: PluginContext,
        topologyBcif: Uint8Array,
        onPick?: (info: AtomInfo | null) => void,
    ): Promise<LiveViewer> {
        const viewer = new LiveViewer(plugin);
        await viewer.build(topologyBcif);
        viewer.subscribeClick(onPick);
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
        await this.setRepresentations([]); // the default (ball-and-stick / element-symbol)
    }

    /**
     * Declaratively set the representations from Python specs. Tears down the
     * current ones and rebuilds; an empty list restores the default. Each spec:
     * { id, type, color?, colorValue?, on?: <index-set>, opacity?, params? }.
     * Representations hang off the live structure, so they coordinate-update per frame.
     */
    async setRepresentations(specs: any[]) {
        if (this.reprNodes.length) {
            const b = this.plugin.state.data.build();
            for (const node of this.reprNodes) if (node.ref) b.delete(node.ref);
            await b.commit();
            this.reprNodes = [];
        }
        const list = specs && specs.length ? specs : [{ type: 'ball-and-stick', color: 'element-symbol' }];
        for (const spec of list) {
            let target: StateObjectSelector = this.structure;
            if (spec.on) {
                const struct = this.currentStructure();
                const indices = decodeIndexSet(spec.on);
                if (!struct || indices.length === 0) continue;
                const bundle = StructureElement.Bundle.fromLoci(lociFromElementIndices(struct, indices));
                const comp = await this.plugin.builders.structure.tryCreateComponent(
                    this.structure,
                    { type: { name: 'bundle', params: bundle }, nullIfEmpty: true, label: `rep:${spec.id}` } as any,
                    `rep-comp:${spec.id}`,
                );
                if (!comp) continue;
                target = comp;
                this.reprNodes.push(comp);
            }
            const params: any = { type: spec.type };
            if (spec.color) params.color = spec.color;
            if (spec.colorValue != null) params.colorParams = { value: decodeColor(spec.colorValue) };
            const typeParams: any = spec.params ? { ...spec.params } : {};
            if (spec.opacity != null) typeParams.alpha = spec.opacity;
            if (Object.keys(typeParams).length) params.typeParams = typeParams;
            const repr = await this.plugin.builders.structure.representation.addRepresentation(target, params);
            this.reprNodes.push(repr);
        }
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
        // A frame rebuilds the structure; cheaply remap the cached highlight loci
        // onto it (O(selected)) instead of rebuilding it from indices.
        this.reapplyHighlight();
    }

    /** Show the selection overlay on the given positional atom indices (empty clears). */
    setHighlight(indices: number[]) {
        const structure = this.currentStructure();
        this.highlightIndices = indices;
        if (!structure || indices.length === 0) {
            this.clearSelection();
            return;
        }
        this.highlightLoci = lociFromElementIndices(structure, indices);
        const selection = this.plugin.managers.structure.selection;
        selection.clear();
        selection.fromLoci('set', this.highlightLoci);
    }

    /** Zoom the camera to the given positional atom indices. */
    focusIndices(indices: number[]) {
        const structure = this.currentStructure();
        if (!structure || indices.length === 0) return;
        this.plugin.managers.camera.focusLoci(lociFromElementIndices(structure, indices));
    }

    /** Clear any active highlight. */
    clearSelection() {
        this.highlightIndices = [];
        this.highlightLoci = undefined;
        this.plugin.managers.structure.selection.clear();
    }

    private currentStructure(): Structure | undefined {
        return this.structure?.obj?.data as Structure | undefined;
    }

    // The frame replaced the structure object, so remap the cached loci onto the
    // new one (O(selected)) and re-apply — no rebuild from indices.
    private reapplyHighlight() {
        if (!this.highlightLoci) return;
        const structure = this.currentStructure();
        if (!structure) return;
        this.highlightLoci = StructureElement.Loci.remap(this.highlightLoci, structure);
        const selection = this.plugin.managers.structure.selection;
        selection.clear();
        selection.fromLoci('set', this.highlightLoci);
    }

    /**
     * Add a measurement primitive from atom-index groups. Mol*'s measurement
     * manager builds these from position-independent bundles that depend on the
     * structure, so they recompute automatically as coordinates stream in.
     */
    async addMeasurement(
        id: string,
        kind: string,
        groups: number[][],
        options: { opacity?: number; label?: boolean; text?: string },
    ) {
        const structure = this.currentStructure();
        if (!structure) return;
        await this.removePrimitive(id); // replace if this id already exists
        const loci = groups.map((g) => lociFromElementIndices(structure, g));
        const m = this.plugin.managers.structure.measurement;
        const opacity = options.opacity ?? 0.35;
        const withText = <T extends string>(base: T[]): T[] => (options.label === false ? base : ([...base, 'text'] as T[]));
        let res: any;
        if (kind === 'distance' && loci.length >= 2) {
            res = await m.addDistance(loci[0], loci[1], {
                visualParams: { visuals: withText(['lines']) as any },
            });
        } else if (kind === 'angle' && loci.length >= 3) {
            res = await m.addAngle(loci[0], loci[1], loci[2], {
                visualParams: { visuals: withText(['vectors', 'sector', 'arc']) as any, sectorOpacity: opacity },
            });
        } else if (kind === 'dihedral' && loci.length >= 4) {
            res = await m.addDihedral(loci[0], loci[1], loci[2], loci[3], {
                visualParams: { visuals: withText(['vectors', 'extenders', 'connector', 'sector']) as any, sectorOpacity: opacity },
            });
        } else if (kind === 'label' && loci.length >= 1) {
            res = await m.addLabel(loci[0], { visualParams: { customText: options.text ?? '' } });
        }
        if (res?.selection) this.primitives.set(id, res.selection);
    }

    /** Remove a single primitive by id. */
    async removePrimitive(id: string) {
        const selection = this.primitives.get(id);
        if (!selection) return;
        this.primitives.delete(id);
        if (selection.ref) await this.plugin.state.data.build().delete(selection.ref).commit();
    }

    /** Remove all primitives. */
    async clearPrimitives() {
        for (const id of Array.from(this.primitives.keys())) await this.removePrimitive(id);
    }

    /**
     * Set the click interaction mode. `'select'` builds a selection reported to
     * Python; `'distance'|'angle'|'dihedral'|'label'` collect N clicks then draw
     * that measurement; `'off'` does neither. Modes are mutually exclusive.
     */
    setClickMode(mode: string) {
        this.clickMode = mode;
        this.measurePending = [];
    }

    private subscribeClick(onPick?: (info: AtomInfo | null) => void) {
        this.pickHandler = onPick;
        this.plugin.behaviors.interaction.click.subscribe((e) => {
            const loci = e.current.loci;
            const location = StructureElement.Loci.is(loci)
                ? StructureElement.Loci.getFirstLocation(loci)
                : undefined;
            if (this.pickHandler) {
                this.pickHandler(location ? {
                    id: StructureProperties.atom.id(location),
                    name: StructureProperties.atom.label_atom_id(location),
                    resname: StructureProperties.atom.label_comp_id(location),
                    resseq: StructureProperties.residue.label_seq_id(location),
                    chain: StructureProperties.chain.label_asym_id(location),
                } : null);
            }
            if (this.clickMode === 'select') this.handleSelectionClick(location, !!e.modifiers?.shift);
            else if (this.clickMode !== 'off') this.handleMeasureClick(location, this.clickMode);
        });
    }

    // Click selects just that atom; shift-click toggles it in the set; a click on
    // empty space clears. Highlights the set and reports it back to Python.
    private handleSelectionClick(location: StructureElement.Location | undefined, shift: boolean) {
        if (!location) {
            if (!shift) this.mouseSelectionSet.clear();
        } else {
            const el = location.element as unknown as number;
            if (shift) {
                if (this.mouseSelectionSet.has(el)) this.mouseSelectionSet.delete(el);
                else this.mouseSelectionSet.add(el);
            } else {
                this.mouseSelectionSet.clear();
                this.mouseSelectionSet.add(el);
            }
        }
        const indices = Array.from(this.mouseSelectionSet).sort((a, b) => a - b);
        this.setHighlight(indices);
        this.onSelectionChange?.(indices);
    }

    // Collect clicks (in order) until we have the arity for `kind`, highlighting
    // progress; then report the atoms to Python, which draws the primitive. A
    // click on empty space resets the in-progress collection.
    private handleMeasureClick(location: StructureElement.Location | undefined, kind: string) {
        const arity = MEASURE_ARITY[kind];
        if (!arity) return;
        if (!location) {
            this.measurePending = [];
            this.setHighlight([]);
            return;
        }
        const el = location.element as unknown as number;
        if (this.measurePending.includes(el)) return; // ignore a repeat click on the same atom
        this.measurePending.push(el);
        this.setHighlight([...this.measurePending].sort((a, b) => a - b));
        if (this.measurePending.length >= arity) {
            const atoms = this.measurePending.slice(); // click order matters (vertex, torsion)
            this.measurePending = [];
            this.setHighlight([]);
            this.onMeasure?.(kind, atoms);
        }
    }
}

const MEASURE_ARITY: Record<string, number> = { distance: 2, angle: 3, dihedral: 4, label: 1 };

/**
 * Build an element loci for the given (sorted) positional atom indices. Looks
 * each index up per unit by binary search — O(selected·log), not a full scan of
 * every element — so it stays cheap for large structures.
 */
function lociFromElementIndices(structure: Structure, indices: number[]): StructureElement.Loci {
    const elements: { unit: Unit; indices: OrderedSet }[] = [];
    for (const unit of structure.units) {
        const us = unit.elements;
        const positions: number[] = [];
        for (let i = 0; i < indices.length; i++) {
            const pos = OrderedSet.indexOf(us, indices[i] as any);
            if (pos >= 0) positions.push(pos);
        }
        if (positions.length) {
            elements.push({ unit, indices: SortedArray.ofSortedArray(positions) });
        }
    }
    return StructureElement.Loci(structure, elements as any);
}

/** Decode a wire index-set: either explicit `{list}` or run-length `{runs}`. */
function decodeIndexSet(enc: any): number[] {
    if (!enc) return [];
    if (enc.runs) {
        const out: number[] = [];
        for (const [s, e] of enc.runs) for (let i = s; i <= e; i++) out.push(i);
        return out;
    }
    return enc.list ?? [];
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
            } else if (msg.type === 'highlight' && viewer) {
                viewer.setHighlight(decodeIndexSet(msg.atoms));
            } else if (msg.type === 'focus' && viewer) {
                viewer.focusIndices(decodeIndexSet(msg.atoms));
            } else if (msg.type === 'representations' && viewer) {
                await viewer.setRepresentations(msg.reprs ?? []);
            } else if (msg.type === 'click-mode' && viewer) {
                viewer.setClickMode(String(msg.mode ?? 'off'));
            } else if (msg.type === 'primitive' && viewer) {
                try {
                    if (msg.action === 'add') {
                        await viewer.addMeasurement(String(msg.id), String(msg.kind), msg.groups ?? [], msg.options ?? {});
                    } else if (msg.action === 'remove') {
                        await viewer.removePrimitive(String(msg.id));
                    } else if (msg.action === 'clear') {
                        await viewer.clearPrimitives();
                    }
                } catch {
                    // ignore malformed primitive commands
                }
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
            viewer.onSelectionChange = (indices) => ws.send(JSON.stringify({ type: 'mouse-selection', indices }));
            viewer.onMeasure = (kind, atoms) => ws.send(JSON.stringify({ type: 'measure', kind, atoms }));
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
