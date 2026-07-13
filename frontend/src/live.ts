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
import { Model, StructureElement, StructureProperties } from 'molstar/lib/mol-model/structure';
import { Coordinates, Frame, Time } from 'molstar/lib/mol-model/structure/coordinates';

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
    private version = 0;
    private nAtoms = 0;

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

const TAG_TOPOLOGY = 0;
const TAG_FRAME = 1;

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
        if (typeof ev.data === 'string') return; // reserved for future control messages
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
