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
import { Bond, Model, Structure, StructureElement, StructureProperties, StructureSelection, Unit } from 'molstar/lib/mol-model/structure';
import { Color, ColorScale } from 'molstar/lib/mol-util/color';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import { ColorThemeCategory } from 'molstar/lib/mol-theme/color/categories';
import { Coordinates, Frame, Time } from 'molstar/lib/mol-model/structure/coordinates';
import { Script } from 'molstar/lib/mol-script/script';
import { transpiler as pymolTranspiler } from 'molstar/lib/mol-script/transpilers/pymol/parser';
import { CustomInteractions, InteractionsShape } from 'molstar/lib/extensions/interactions/transforms';
import { ShapeRepresentation3D } from 'molstar/lib/mol-plugin-state/transforms/representation';
import { OrderedSet, SortedArray } from 'molstar/lib/mol-data/int';
import type { Canvas3DProps } from 'molstar/lib/mol-canvas3d/canvas3d';
import { decodeColor } from 'molstar/lib/mol-util/color/utils';
import { Shape } from 'molstar/lib/mol-model/shape';
import { Points } from 'molstar/lib/mol-geo/geometry/points/points';
import { PointsBuilder } from 'molstar/lib/mol-geo/geometry/points/points-builder';
import { Lines } from 'molstar/lib/mol-geo/geometry/lines/lines';
import { LinesBuilder } from 'molstar/lib/mol-geo/geometry/lines/lines-builder';

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

// -- probe2 contact-dot surface ------------------------------------------
//
// A probe2 "dotkin": a point cloud of contact dots plus line "spikes" for the
// overlaps, both at raw model coordinates and coloured MolProbity-style. Two
// custom state transforms turn the parsed arrays into Mol* Shapes (Points and
// Lines), each rendered by ShapeRepresentation3D.

/** Points shape from dot positions (`xyz` interleaved) + per-dot packed rgb. */
function buildDotPoints(xyz: Float32Array, rgb: Uint32Array): Shape<Points> {
    const n = rgb.length;
    const builder = PointsBuilder.create(n, 1);
    for (let i = 0; i < n; i++) builder.add(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2], i);
    return Shape.create(
        'probe-dots', {}, builder.getPoints(),
        (g) => Color(rgb[g]), () => 1, () => 'probe dot',
    );
}

/** Lines shape from spike segments (`start`/`end` interleaved) + per-spike rgb. */
function buildDotLines(starts: Float32Array, ends: Float32Array, rgb: Uint32Array): Shape<Lines> {
    const n = rgb.length;
    const builder = LinesBuilder.create(n, 1);
    for (let i = 0; i < n; i++) {
        builder.add(starts[i * 3], starts[i * 3 + 1], starts[i * 3 + 2],
                    ends[i * 3], ends[i * 3 + 1], ends[i * 3 + 2], i);
    }
    return Shape.create(
        'probe-spikes', {}, builder.getLines(),
        (g) => Color(rgb[g]), () => 1, () => 'clash spike',
    );
}

const ProbeDotsPoints = PluginStateTransform.BuiltIn({
    name: 'pxviewer-probe-points',
    display: { name: 'Probe Dots' },
    from: SO.Root,
    to: SO.Shape.Provider,
    params: {
        xyz: PD.Value<Float32Array>(new Float32Array(0), { isHidden: true }),
        rgb: PD.Value<Uint32Array>(new Uint32Array(0), { isHidden: true }),
        sizeFactor: PD.Value<number>(2.5, { isHidden: true }),
    },
})({
    apply({ params }) {
        const p = params as { xyz: Float32Array; rgb: Uint32Array; sizeFactor: number };
        return new SO.Shape.Provider({
            label: 'Probe Dots',
            // Small round dots at a physical (world-scaled) size, not the default
            // 3-unit fixed squares.
            data: p,
            params: PD.withDefaults(Points.Params, {
                pointStyle: 'circle',
                // Constant small screen-space dots (classic probe/kinemage look).
                // Attenuation would scale gl_PointSize by (viewportH/2)/-z * 5, which
                // makes each dot explode to ~200px as you zoom in — the opposite of
                // what a dense contact surface wants. Validation markers pass a larger
                // sizeFactor so a sparse handful stays prominent.
                pointSizeAttenuation: false,
                sizeFactor: p.sizeFactor,
            }),
            getShape: (_ctx, data) => buildDotPoints(data.xyz, data.rgb),
            geometryUtils: Points.Utils,
        }, { label: 'Probe Dots' });
    },
});
type ProbeDotsPoints = typeof ProbeDotsPoints;

const ProbeDotsLines = PluginStateTransform.BuiltIn({
    name: 'pxviewer-probe-lines',
    display: { name: 'Probe Spikes' },
    from: SO.Root,
    to: SO.Shape.Provider,
    params: {
        starts: PD.Value<Float32Array>(new Float32Array(0), { isHidden: true }),
        ends: PD.Value<Float32Array>(new Float32Array(0), { isHidden: true }),
        rgb: PD.Value<Uint32Array>(new Uint32Array(0), { isHidden: true }),
    },
})({
    apply({ params }) {
        const p = params as { starts: Float32Array; ends: Float32Array; rgb: Uint32Array };
        return new SO.Shape.Provider({
            label: 'Probe Spikes',
            data: p,
            params: PD.withDefaults(Lines.Params, {}),
            getShape: (_ctx, data) => buildDotLines(data.starts, data.ends, data.rgb),
            geometryUtils: Lines.Utils,
        }, { label: 'Probe Spikes' });
    },
});
type ProbeDotsLines = typeof ProbeDotsLines;

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

// -- colour by per-atom attribute ----------------------------------------
//
// A custom colour theme driven by a Python-supplied per-atom scalar array
// (indexed by i_seq == model element index == wire index). Values are mapped
// through a Mol* ColorScale; non-finite values take the missing colour. This one
// theme handles b-factor, occupancy and any arbitrary attribute uniformly.

const ATTRIBUTE_MISSING_COLOR = Color(0x808080);

const AttributeColorThemeParams = {
    values: PD.Value<ArrayLike<number>>([], { isHidden: true }),
    domain: PD.Interval([0, 1]),
    palette: PD.Value<any>('turbo', { isHidden: true }),
    missing: PD.Color(ATTRIBUTE_MISSING_COLOR),
};

function attributeColorTheme(_ctx: any, props: any) {
    const scale = ColorScale.create({ domain: props.domain, listOrName: props.palette });
    const values = props.values as ArrayLike<number>;
    const missing = props.missing;
    const pick = (i: number) => {
        const v = values[i];
        return v == null || Number.isNaN(v) ? missing : scale.color(v);
    };
    function color(location: any) {
        if (StructureElement.Location.is(location)) return pick(location.element as unknown as number);
        if (Bond.isLocation(location)) return pick(location.aUnit.elements[location.aIndex] as unknown as number);
        return missing;
    }
    return {
        factory: attributeColorTheme,
        granularity: 'group' as const,
        preferSmoothing: true,
        color,
        props,
        description: 'Colour by a pxviewer per-atom attribute.',
        legend: scale.legend,
    };
}

const AttributeColorThemeProvider: any = {
    name: 'pxviewer-attribute',
    label: 'pxviewer Attribute',
    category: ColorThemeCategory.Misc,
    factory: attributeColorTheme,
    getParams: () => AttributeColorThemeParams,
    defaultValues: PD.getDefaultValues(AttributeColorThemeParams),
    isApplicable: () => true,
};

const attributeThemeRegistered = new WeakSet<PluginContext>();

/** Register the pxviewer per-atom-attribute colour theme on a plugin (once). */
export function registerAttributeColorTheme(plugin: PluginContext) {
    if (attributeThemeRegistered.has(plugin)) return;
    plugin.representation.structure.themes.colorThemeRegistry.add(AttributeColorThemeProvider);
    attributeThemeRegistered.add(plugin);
}

/** Resolve a wire palette (a Mol* colour-list name, or explicit colours) for the scale. */
function resolvePalette(palette: any): any {
    if (Array.isArray(palette)) return palette.map((c) => decodeColor(c));
    return palette; // a ColorListName string
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
    private interactionsNode: StateObjectSelector | undefined;
    private clashesNode: StateObjectSelector | undefined;
    private probeChannels: Map<number, StateObjectSelector[]> = new Map();
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
            if (spec.color === 'attribute' && spec.attribute?.resolved) {
                // Values arrive on the binary attribute channel and are attached as
                // `resolved` (a Float32Array) by the connection before we get here.
                const a = spec.attribute;
                params.color = 'pxviewer-attribute';
                params.colorParams = { values: a.resolved, domain: a.domain, palette: resolvePalette(a.palette) };
            } else if (spec.color && spec.color !== 'attribute') {
                params.color = spec.color;
            }
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
        this.applyOwnSelection(this.highlightLoci);
    }

    /**
     * Apply `loci` as *this structure's* contribution to the plugin-global
     * selection, leaving other structures (other models in a multi-model scene)
     * untouched. We remove this structure's whole loci first — that clears only
     * our previous highlight, and since the current selection here is just that
     * small highlight, 'remove' is O(selected), not O(atoms) — then add the new
     * one. Using 'set' instead would wipe every other model's selection too.
     */
    private applyOwnSelection(loci: StructureElement.Loci | undefined) {
        const structure = this.currentStructure();
        if (!structure) return;
        const selection = this.plugin.managers.structure.selection;
        selection.fromLoci('remove', Structure.toStructureElementLoci(structure));
        if (loci) selection.fromLoci('add', loci);
    }

    /** Zoom the camera to the given positional atom indices. */
    focusIndices(indices: number[]) {
        const structure = this.currentStructure();
        if (!structure || indices.length === 0) return;
        this.plugin.managers.camera.focusLoci(lociFromElementIndices(structure, indices));
    }

    /**
     * Aim the camera at `target` with an explicit orientation: `up` is screen-up and
     * `direction` is the view axis (eye -> target). Used to frame a residue with its
     * N->C backbone left-to-right and side chain up.
     */
    orient(target: number[], up: number[], direction: number[], radius: number) {
        const camera = this.plugin.canvas3d?.camera;
        if (!camera) return;
        // getInvariantFocus sets up/dir absolutely; camera.focus() instead runs them
        // through matchDirection (flips to stay near the current view), which would
        // not honour the requested orientation.
        const snapshot = camera.getInvariantFocus(
            Vec3.create(target[0], target[1], target[2]),
            radius,
            Vec3.create(up[0], up[1], up[2]),
            Vec3.create(direction[0], direction[1], direction[2]),
        );
        camera.setState(snapshot, 250);
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
            this.highlightLoci = loci;
            this.applyOwnSelection(indices.length ? loci : undefined);
        }
        if (opts.focus && indices.length) {
            this.plugin.managers.camera.focusLoci(loci);
        }
        return indices;
    }

    /** Clear any active highlight. */
    clearSelection() {
        this.highlightIndices = [];
        this.highlightLoci = undefined;
        this.applyOwnSelection(undefined);
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
        this.applyOwnSelection(this.highlightLoci);
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
     * Draw an explicit set of non-covalent interactions supplied by Python as
     * typed atom-index pairs (e.g. hydrogen-bond between atoms 0 and 1). Unlike
     * the computed overlay, nothing is inferred — these are exactly the contacts
     * given. They hang off the live structure via `CustomInteractions`, so their
     * endpoints track streamed coordinates. An empty list clears them.
     *
     * `contacts` items: { kind, a, b, description? } where a/b are positional
     * atom indices (Mol*'s `atom_index` == our source-index identity).
     */
    async setInteractions(contacts: { kind: string; a: number; b: number; description?: string }[]) {
        if (!contacts || contacts.length === 0) {
            await this.clearInteractions();
            return;
        }
        const ref = this.structure.ref;
        // Each endpoint is a single atom addressed by its source index; both sides
        // live in the same live structure, so a/bStructureRef are the same ref.
        const interactions = contacts.map((c) => ({
            kind: c.kind,
            aStructureRef: ref,
            a: { atom_index: c.a },
            bStructureRef: ref,
            b: { atom_index: c.b },
            description: c.description,
        }));
        if (this.interactionsNode?.ref) {
            await this.plugin.state.data
                .build()
                .to(this.interactionsNode)
                .update((old: any) => ({ ...old, interactions }))
                .commit();
            return;
        }
        const node = this.plugin.state.data
            .build()
            .toRoot()
            .apply(CustomInteractions, { interactions } as any, { dependsOn: [ref] });
        node.apply(InteractionsShape).apply(ShapeRepresentation3D);
        this.interactionsNode = node.selector;
        await node.commit();
    }

    /** Remove the explicit interactions overlay, if any. */
    async clearInteractions() {
        const node = this.interactionsNode;
        this.interactionsNode = undefined;
        if (node?.ref) await this.plugin.state.data.build().delete(node.ref).commit();
    }

    /**
     * Draw steric clashes supplied by Python as atom-index pairs. Rendered as
     * distinct red solid cylinders (visually separate from the dashed interaction
     * notation) via a dedicated `CustomInteractions` node, so — like interactions
     * — the markers track streamed coordinates. An empty list clears them.
     *
     * Clashes reuse the interactions shape but on their own node, with the marker
     * kind restyled red/solid; Mol* has no general clash detector of its own, so
     * the pairs are exactly the ones Python computed.
     */
    async setClashes(pairs: { a: number; b: number }[]) {
        if (!pairs || pairs.length === 0) {
            await this.clearClashes();
            return;
        }
        const ref = this.structure.ref;
        // Marker kind is arbitrary (this node renders only 'unknown', restyled as a
        // clash); the identity that matters is the atom_index pair.
        const interactions = pairs.map((p) => ({
            kind: 'unknown',
            aStructureRef: ref,
            a: { atom_index: p.a },
            bStructureRef: ref,
            b: { atom_index: p.b },
        }));
        if (this.clashesNode?.ref) {
            await this.plugin.state.data
                .build()
                .to(this.clashesNode)
                .update((old: any) => ({ ...old, interactions }))
                .commit();
            return;
        }
        const node = this.plugin.state.data
            .build()
            .toRoot()
            .apply(CustomInteractions, { interactions } as any, { dependsOn: [ref] });
        node.apply(InteractionsShape, {
            kinds: ['unknown'],
            styles: { unknown: { color: CLASH_COLOR, style: 'solid', radius: 0.1 } },
        } as any).apply(ShapeRepresentation3D);
        this.clashesNode = node.selector;
        await node.commit();
    }

    /** Remove the clash overlay, if any. */
    async clearClashes() {
        const node = this.clashesNode;
        this.clashesNode = undefined;
        if (node?.ref) await this.plugin.state.data.build().delete(node.ref).commit();
    }

    /**
     * Draw a probe2 contact-dot surface from a flat dot buffer: each dot is
     * `[loc xyz][spike xyz][rgb]`. All dots become a point cloud; the ones whose
     * spike differs from their location (the overlaps) also get a line spike.
     */
    async setProbeDots(buffer: ArrayBuffer, offset: number) {
        const dv = new DataView(buffer);
        // [u32 channel][u32 n][dots...] — channel keeps overlays (contacts/clashes)
        // independent so they toggle separately.
        const channel = dv.getUint32(offset, true);
        await this.clearProbeDots(channel);
        const n = dv.getUint32(offset + 4, true);
        let p = offset + 8;
        const locs = new Float32Array(n * 3);
        const rgb = new Uint32Array(n);
        // Spikes: collected only for overlaps (loc !== spike).
        const spikeStart: number[] = [];
        const spikeEnd: number[] = [];
        const spikeRgb: number[] = [];
        for (let i = 0; i < n; i++) {
            const lx = dv.getFloat32(p, true), ly = dv.getFloat32(p + 4, true), lz = dv.getFloat32(p + 8, true);
            const sx = dv.getFloat32(p + 12, true), sy = dv.getFloat32(p + 16, true), sz = dv.getFloat32(p + 20, true);
            const c = dv.getUint32(p + 24, true);
            p += 28;
            locs[i * 3] = lx; locs[i * 3 + 1] = ly; locs[i * 3 + 2] = lz;
            rgb[i] = c;
            if (lx !== sx || ly !== sy || lz !== sz) {
                spikeStart.push(lx, ly, lz); spikeEnd.push(sx, sy, sz); spikeRgb.push(c);
            }
        }
        // Validation markers (channels >= 10) are a sparse handful, so draw them
        // large; probe2 contact/clash surfaces (channels 0/1) stay small.
        const sizeFactor = channel >= VALIDATION_CHANNEL_BASE ? 16 : 2.5;
        const nodes: StateObjectSelector[] = [];
        const build = this.plugin.state.data.build();
        const pts = build.toRoot().apply(ProbeDotsPoints, { xyz: locs, rgb, sizeFactor }).apply(ShapeRepresentation3D);
        nodes.push(pts.selector);
        if (spikeRgb.length) {
            const lines = build.toRoot().apply(ProbeDotsLines, {
                starts: new Float32Array(spikeStart),
                ends: new Float32Array(spikeEnd),
                rgb: new Uint32Array(spikeRgb),
            }).apply(ShapeRepresentation3D);
            nodes.push(lines.selector);
        }
        await build.commit();
        this.probeChannels.set(channel, nodes);
    }

    /** Remove a probe dot overlay: one `channel`, or all when omitted. */
    async clearProbeDots(channel?: number) {
        const channels = channel === undefined ? [...this.probeChannels.keys()] : [channel];
        const b = this.plugin.state.data.build();
        let any = false;
        for (const ch of channels) {
            const nodes = this.probeChannels.get(ch);
            if (!nodes) continue;
            for (const node of nodes) if (node.ref) b.delete(node.ref);
            this.probeChannels.delete(ch);
            any = true;
        }
        if (any) await b.commit();
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
            // The click behaviour is plugin-global, so in a multi-structure scene
            // every viewer is notified. A click that landed on an atom belongs to
            // exactly one structure — only that structure's viewer responds. (An
            // empty-space click has no structure, so all viewers see it.)
            if (StructureElement.Loci.is(loci)) {
                const own = this.currentStructure();
                if (!own || !Structure.areRootsEquivalent(loci.structure, own)) return;
            }
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

/** Positional atom rows (model element indices) covered by an element loci. */
function collectElementIndices(loci: StructureElement.Loci): number[] {
    const set = new Set<number>();
    StructureElement.Loci.forEachLocation(loci, (loc) => { set.add(loc.element as unknown as number); });
    return Array.from(set).sort((a, b) => a - b);
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
const TAG_ATTRIBUTE = 2;
const TAG_DOTS = 3;
// Dot channels >= this are validation markers (drawn large); must match
// pxviewer.validation.CHANNEL_BASE.
const VALIDATION_CHANNEL_BASE = 10;

const INTERACTIONS_TAG = 'pxviewer-interactions';
const CLASH_COLOR = 0xee2222; // red — reads as "bad contact"

/**
 * Show or hide the *computed* "Non-covalent Interactions" representation on every
 * structure in the scene — whether it was loaded from an MVSJ scene/file or is
 * streaming live. Mol* infers the contacts. We tag the representations we add so
 * we can find and remove exactly those, and skip structures that already have one
 * (so a repeated toggle is a no-op). Hanging the representation off each structure
 * means it recomputes per frame. For explicit, Python-supplied contacts instead,
 * see `LiveViewer.setInteractions`.
 */
async function setComputedInteractions(plugin: PluginContext, visible: boolean) {
    const structures = plugin.state.data.selectQ((q: any) => q.rootsOfType(SO.Molecule.Structure));
    if (visible) {
        for (const cell of structures) {
            const ref = cell.transform.ref;
            const existing = plugin.state.data.selectQ((q: any) => q.byRef(ref).subtree().withTag(INTERACTIONS_TAG));
            if (existing.length) continue;
            // 'interactions' is a computed/extension type, so it's outside the
            // built-in props union; the registered provider resolves it at runtime.
            await plugin.builders.structure.representation.addRepresentation(
                ref,
                { type: 'interactions' } as any,
                { tag: INTERACTIONS_TAG },
            );
        }
    } else {
        const reprs = plugin.state.data.selectQ((q: any) => q.root.subtree().withTag(INTERACTIONS_TAG));
        if (!reprs.length) return;
        const b = plugin.state.data.build();
        for (const cell of reprs) b.delete(cell.transform.ref);
        await b.commit();
    }
}

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
    registerAttributeColorTheme(plugin);
    setAxis(plugin, false);  // XYZ axes off by default; the Settings toggle turns them on
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    let viewer: LiveViewer | null = null;
    let building = false;
    // Per-atom attribute values (colour-by-attribute), received as binary and
    // referenced by key from representation specs. Held independent of the viewer,
    // since they may arrive while it is still building.
    const attributeValues = new Map<string, Float32Array>();

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

    // Control messages that act on the LiveViewer. On connect the server replays
    // state (representations, highlight, …) right after the topology, which can
    // arrive while the viewer is still building asynchronously — so these are
    // queued until the viewer exists, then flushed in order.
    const VIEWER_MSG_TYPES = new Set([
        'interactions', 'clashes', 'highlight', 'focus', 'orient', 'representations', 'click-mode', 'primitive', 'select', 'dots',
    ]);
    const pendingControl: any[] = [];
    let pendingDots: ArrayBuffer[] = [];  // dot buffers (per channel) that beat the viewer build

    const handleControlMessage = async (msg: any) => {
            if (msg.type === 'axis' && typeof msg.visible === 'boolean') {
                await setAxis(plugin, msg.visible);
            } else if (msg.type === 'reset-view') {
                plugin.managers.camera.reset();  // reframe the whole scene, default orientation
            } else if (msg.type === 'computed-interactions' && typeof msg.visible === 'boolean') {
                await setComputedInteractions(plugin, msg.visible);
            } else if (msg.type === 'interactions' && viewer) {
                if (msg.action === 'clear') await viewer.clearInteractions();
                else await viewer.setInteractions(msg.contacts ?? []);
            } else if (msg.type === 'clashes' && viewer) {
                if (msg.action === 'clear') await viewer.clearClashes();
                else await viewer.setClashes(msg.pairs ?? []);
            } else if (msg.type === 'dots' && viewer) {
                if (msg.action === 'clear') await viewer.clearProbeDots(msg.channel ?? undefined);
            } else if (msg.type === 'volume_color' && typeof msg.ref === 'string' && typeof msg.color === 'string') {
                await setVolumeColor(plugin, msg.ref, msg.color);
            } else if (msg.type === 'volume_opacity' && typeof msg.ref === 'string' && typeof msg.opacity === 'number') {
                await setVolumeOpacity(plugin, msg.ref, msg.opacity);
            } else if (msg.type === 'volume_style' && typeof msg.ref === 'string' && typeof msg.style === 'string') {
                await setVolumeStyle(plugin, msg.ref, msg.style);
            } else if (msg.type === 'volume_position' && typeof msg.ref === 'string' && Array.isArray(msg.position) && msg.position.length === 3) {
                await setVolumePosition(plugin, msg.ref, msg.position);
            } else if (msg.type === 'highlight' && viewer) {
                viewer.setHighlight(decodeIndexSet(msg.atoms));
            } else if (msg.type === 'focus' && viewer) {
                viewer.focusIndices(decodeIndexSet(msg.atoms));
            } else if (msg.type === 'orient' && viewer) {
                viewer.orient(msg.target, msg.up, msg.direction, msg.radius);
            } else if (msg.type === 'representations' && viewer) {
                // Attach the per-atom values (received on the binary attribute
                // channel) to any attribute-coloured spec before applying.
                const reprs = (msg.reprs ?? []).map((r: any) =>
                    r.color === 'attribute' && r.attribute
                        ? { ...r, attribute: { ...r.attribute, resolved: attributeValues.get(r.attribute.key) } }
                        : r
                );
                await viewer.setRepresentations(reprs);
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
            } else if (msg.type === 'select' && viewer) {
                // Resolve a PyMOL selection in the viewer and echo the matched
                // atom indices back so Python knows what was selected.
                let indices: number[] = [];
                let error: string | undefined;
                try {
                    indices = viewer.applySelection(String(msg.expression ?? ''), { highlight: !!msg.highlight, focus: !!msg.focus });
                } catch (e) {
                    error = e instanceof Error ? e.message : String(e);
                }
                ws.send(JSON.stringify({ type: 'selection-result', reqId: msg.reqId, indices, error }));
            }
    };

    ws.onmessage = async (ev) => {
        if (typeof ev.data === 'string') {
            // Server -> client control messages (JSON text).
            let msg: any;
            try { msg = JSON.parse(ev.data); } catch { return; }
            // Defer viewer-dependent messages that beat the (async) viewer build.
            if (VIEWER_MSG_TYPES.has(msg.type) && !viewer) {
                pendingControl.push(msg);
                return;
            }
            await handleControlMessage(msg);
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
            // Now that the viewer exists, apply anything that arrived while building.
            const queued = pendingControl.splice(0);
            for (const m of queued) await handleControlMessage(m);
            for (const buf of pendingDots.splice(0)) await viewer.setProbeDots(buf, 4);
        } else if (tag === TAG_FRAME && viewer) {
            // [u32 tag][u32 frameIndex][f32 * 3N]; coordinates start at byte 8.
            const coords = new Float32Array(buffer, 8);
            await viewer.update(coords);
        } else if (tag === TAG_ATTRIBUTE) {
            // [u32 tag][u32 keyLen][key utf8][pad to 4][f32 * N]. Stored regardless
            // of viewer state (it may still be building); applied when the matching
            // representation is processed. NaN = missing.
            const dv = new DataView(buffer);
            const keyLen = dv.getUint32(4, true);
            const key = new TextDecoder().decode(new Uint8Array(buffer, 8, keyLen));
            const valuesOffset = 8 + keyLen + ((4 - (keyLen % 4)) % 4);
            attributeValues.set(key, new Float32Array(buffer, valuesOffset));
        } else if (tag === TAG_DOTS) {
            // [u32 tag][u32 channel][u32 n][per dot: 6 f32 (loc, spike) + u32 rgb].
            // Buffer per channel if the viewer is still building (dots are not a
            // droppable frame).
            if (viewer) await viewer.setProbeDots(buffer, 4);
            else pendingDots.push(buffer);
        }
    };

    return { close: () => ws.close() };
}
