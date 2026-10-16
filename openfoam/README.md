# PR-EnKF for OpenFOAM

OpenFOAM implementation of the PR-EnKF, in which every ensemble member is a full simulation. It infers the parameters of an OpenFOAM model jointly over several simulation domains, and each parameter can take a different value per cluster (flow regime). For the method, see the [tutorial](../tutorials/01_channel_flow.ipynb) and [`docs/theory.md`](../docs/theory.md).

The code targets **OpenFOAM 10** (openfoam.org).

## A general-purpose tool

The EnKF never links to the solver. It interacts with it **only through files**:

* it **writes** the parameters into a parameter file of each member (e.g. `constant/turbulenceProperties`);
* it **reads** the fields and the values sampled at the observation points.

Any OpenFOAM solver can therefore be used (turbulence, transport, chemistry, …), as long as it reads its parameters from that file. If localisation is enabled, the solver must also write the distance of each cell to each cluster (see [Forward-model contract](#forward-model-contract)).

## How it works

* **Domains.** Each folder `sim1 … simN` is an independent case, with its own mesh, boundary conditions and observations. All domains share the same parameter vector.
* **Ensemble.** Each domain gets `nEnsembles` members. Member *i* has the same parameters in every domain.
* **Clusters.** Each parameter has `nClusters` values. The solver decides which cluster each cell belongs to.
* **Cycles.** In cycle *c*, the members of domain *k* run until iteration *c* × `updateInterval[k]`, are sampled at the observation points, and one joint analysis updates the parameters of all members. Members restart from their latest state in the next cycle.

Only the parameters are updated, with the PR-EnKF analysis described in [`docs/theory.md`](../docs/theory.md), with $\mathbf{C}_{dd} = $ `obsUncertainty`$^2$ and $C_{pp,ii}$ set by `priorUncertainty`. In addition to the Python version, the OpenFOAM implementation includes:

* **optional cluster localisation**: for cluster $c$, the gain of each observation is multiplied by $w_c = \exp(-\bar{d}_c^2 / 2L^2)$, where $\bar{d}_c$ is the ensemble-mean distance of the observation point to cluster $c$;
* **parameter bounds**: after the update, values outside $[0.01, 3] \times$ `default` are resampled near the bound, before inflation.

## Repository content and build

```
openfoam/
├── src/
│   ├── EnKF_multiVariable_multiDomain_clustering.H   class declaration
│   ├── EnKF_multiVariable_multiDomain_clustering.C   all EnKF logic
│   ├── EnKFControls.H                                reads and validates system/EnKFDict
│   └── Make/
├── application/
│   ├── EnKF_FoamMultiVariable_multiDomain_clustering/          setup application
│   └── EnKF_FoamMultiVariable_multiDomain_clustering_Update/   update application
├── templates/        EnKFDict, sampleDict, sub.sh, sub_update.sh, submit_and_update.sh
└── example/          a single-building domain
```

The two applications are thin `main()` wrappers around the class in `src/`:

| Binary | When | What it does |
|---|---|---|
| `EnKF_FoamMultiVariable_multiDomain_clustering_v10` | once, with `iterationCount 1` | creates the members, perturbs parameters and observations, decomposes the members (if parallel), writes the SLURM scripts |
| `EnKF_FoamMultiVariable_multiDomain_clustering_Update_v10` | once per cycle | reads all members at the current checkpoint, computes the analysis, writes the updated parameters into every member |

Build both with your OpenFOAM 10 environment loaded:

```bash
cd application/EnKF_FoamMultiVariable_multiDomain_clustering && wmake
cd ../EnKF_FoamMultiVariable_multiDomain_clustering_Update && wmake
```

The binaries are installed in `$FOAM_USER_APPBIN`. Rebuild both after any change in `src/`, since each one compiles the class.

## Setting up a run

### Master case

The master case is a valid OpenFOAM case that holds the EnKF settings and the domains. Both applications call `createTime`/`createMesh`, so it needs a mesh, although nothing is solved on it; copying one domain's `constant/polyMesh` is enough.

```
<case>/                     run everything from here
├── system/
│   ├── controlDict         any valid one
│   ├── fvSchemes, fvSolution
│   ├── EnKFDict            all EnKF settings
│   └── sampleDict          only its 'fields' entry is read: the assimilated fields
├── constant/polyMesh/      any mesh
├── sim1/ … simN/           the domains
├── sub.sh                  SLURM job running the setup binary
├── sub_update.sh           SLURM job running the update binary
└── submit_and_update.sh    driver: setup + all cycles
```

In `sub.sh` and `sub_update.sh`, the EnKF binary must be the **last command**, so that a fatal error marks the job as failed and stops the driver. In `sub_update.sh`, its output must be redirected to a `log.<name>` file on a line containing `Update`; the driver renames it to `log.<name>_cycleN` every cycle. See [`templates/`](templates).

### Domains

Each `simK` is a complete, runnable case of the forward model with its own observations. At setup it is copied into every member.

* Folders are named `sim` followed by an integer, directly in the master case, and sorted numerically. Per-domain lists in `EnKFDict` follow this order. Gaps are allowed; other names are ignored.
* **May differ between domains:** mesh, geometry, boundary conditions, number and position of observation points, processor count, `updateInterval`, localisation radius.
* **Must be identical:** parameter names, `nClusters`, and the assimilated fields.

```
simK/
├── 0/
├── constant/
│   ├── polyMesh/
│   ├── <parameterFile>             parameters, as lists of nClusters values
│   └── experimentalData_<field>    one per assimilated field
└── system/
    ├── controlDict
    ├── fvSchemes, fvSolution
    ├── sampleDict                  observation points and sampled fields
    ├── decomposeParDict            if parallelRun true
    └── EnKFDict                    copy of the master one
```

## Forward-model contract

**The solver must read**

1. its parameters from `<parameterFile>`, one list of `nClusters` values per parameter, on a single line (e.g. `betaStar (0.09 0.09 0.09);`); reading it once at start-up is enough, since members restart every cycle;
2. its own cluster assignment per cell (k-means, VQPCA, …): value *c* of each list applies to the cells assigned to cluster *c*. The EnKF does not need to know the rule.

**The solver must write, at every checkpoint**

| Output | When |
|---|---|
| every assimilated field | always |
| a field named exactly `U` | always (the cell count of each domain is read from it) |
| `clusterDist0 … clusterDist{N−1}` (`volScalarField`, `AUTO_WRITE`) | if `featureSpaceLocalisation true` |

The cluster distances are the distance $d \ge 0$ of each cell to each cluster, not squared, in the metric the solver uses to assign cells (e.g. the square root of the reconstruction error for VQPCA, the Euclidean distance to the centroid for k-means). The prefix is set by `clusterDistanceField`.

**The solver must**

* write ASCII, uncompressed (`writeFormat ascii; writeCompression off;`);
* use time = iteration count, with integer time directories (`deltaT 1` for a steady solver);
* restart from the latest time (`startFrom latestTime;`), also from `processor*` directories when run in parallel;
* run with `-parallel` and work with `decomposePar` / `reconstructPar -latestTime`;
* exit with a non-zero code on failure;
* be on the `PATH` after `envScript` is sourced, under the name `solverExecutable`. Custom libraries are loaded through `libs (...)` in each domain's `controlDict`.

Sampling is done afterwards with `postProcess -func sampleDict -latestTime`, so the solver does not need to sample anything itself.

## Input files

### `system/EnKFDict` (master case)

| Key | Default | Meaning |
|---|---|---|
| `iterationCount` | 0 | current cycle; set to 1 for a fresh start (incremented by the driver) |
| `updateInterval` | 40 | solver iterations per cycle: one value, or one per domain, e.g. `(4 8)` |
| `nCycles` | required | number of analysis cycles |
| `nEnsembles` | 40 | members per domain |
| `obsUncertainty` | 0.02 | absolute standard deviation of the observation error, common to all observations |
| `inflationFactor` | 1.2 | multiplicative inflation of the parameter spread (1 = none) |
| `parallelRun` | true | decompose the members and run them with `-parallel` |
| `cluster { }` | | SLURM settings: `scheduler slurm;`, `account`, `partition`, `timeLimit` (one cycle of all members), `solverExecutable`, `envScript` |
| `parameterFile` | `constant/turbulenceProperties` | parameter file, relative to each member |
| `nClusters` | 1 | values per parameter |
| `parameters` | required | one `{ name; default; perturbRange; priorUncertainty; }` per parameter; each of the last three is a scalar or a list of `nClusters` values |
| `featureSpaceLocalisation` | false | enables cluster localisation |
| `featureSpaceLocalisationRadius` | required if on | $L$: one value, or one per domain, in the units of the solver's distance |
| `clusterDistanceField` | `clusterDist` | prefix of the distance fields |
| `sampledFieldTypes` | | one `{ name; type; component; }` per assimilated field; `type` is `scalar` or `vector`, `component` (0, 1, 2) is the assimilated component |
| `featureSampleSetName` | `enkfPoints` | name of the sample set (and of the `.xy` file) |

For each parameter, `default` is the literature-consistent value $p_i$ (strictly positive; it also sets the bounds), `perturbRange` the relative spread of the initial ensemble, and `priorUncertainty` the relative uncertainty $\gamma_i$ of the pseudo-observations. The two spreads are normally equal.

```
iterationCount  1;
updateInterval  (4 8);
nCycles         10;
nEnsembles      40;
obsUncertainty  0.4;
inflationFactor 1.05;
parallelRun     true;

cluster
{
    scheduler        slurm;
    account          <account>;
    partition        <partition>;
    timeLimit        "12:00:00";
    solverExecutable simpleFoam;
    envScript        "<path to OpenFOAM 10 environment script>";
}

parameterFile   "constant/turbulenceProperties";
nClusters       5;

parameters
(
    { name a1;          default 0.31;  perturbRange 0.1; priorUncertainty 0.1; }
    { name alphaOmega2; default 0.856; perturbRange 0.1; priorUncertainty 0.1; }
    { name betaStar;    default 0.09;  perturbRange 0.1; priorUncertainty 0.1; }
);

featureSpaceLocalisation        true;
featureSpaceLocalisationRadius  (4.3);
clusterDistanceField            clusterDist;

sampledFieldTypes
(
    { name U; type vector; component 0; }
);

featureSampleSetName  enkfPoints;
```

### `system/sampleDict`

In the **master case**, only `fields` is read: it lists the assimilated fields, e.g. `fields (U);`.

In **each domain**, the full dictionary is used by `postProcess`:

```
type                sets;
libs                ("libsampling.so");
interpolationScheme cellPoint;
setFormat           raw;                 // required

sets
(
    enkfPoints                           // = featureSampleSetName
    {
        type    points;
        ordered yes;
        axis    xyz;                     // required
        points
        (
            (-0.04 0 0.130036)           // same order as experimentalData_*
            (-0.04 0 0.13978)
        );
    }
);

fields  (U "clusterDist.*");             // assimilated fields (+ distances if localisation)
```

Columns are found by name in the header of the `.xy` file (`U_x`, `U_y`, `U_z` for vectors, the field name for scalars), so the field order does not matter and extra fields are ignored.

### `constant/experimentalData_<field>` (each domain)

```
experimentalPoints
(
    -0.04  0  0.130036  1.265
    -0.04  0  0.13978   4.118
);
```

* One row per observation point: `x y z value`, where `value` is the assimilated component (for `U` with component 0, $U_x$).
* Rows must be **in the same order as the `sampleDict` points**: points are matched by order, not by coordinates.
* All `experimentalData_*` files of a domain must have the same number of rows; different domains may differ.

### `constant/<parameterFile>` (each domain)

```
kOmegaSSTCoeffs
{
    a1           (0.31 0.31 0.31 0.31 0.31);
    alphaOmega2  (0.856 0.856 0.856 0.856 0.856);
    betaStar     (0.09 0.09 0.09 0.09 0.09);
}
```

* One line per parameter: name, whitespace, a list of `nClusters` values, `;`.
* Each name must appear only once in the file as a whole word, comments included.
* A scalar entry (`betaStar 0.09;`) is converted to a list at setup.

### `system/controlDict` (each domain)

```
startFrom        latestTime;
stopAt           endTime;        // must come before endTime
endTime          4;              // integer; overwritten every cycle
deltaT           1;              // time = iteration
writeControl     timeStep;       // overwritten at setup
writeInterval    4;              // overwritten at setup with updateInterval
writeFormat      ascii;
writeCompression off;
libs             ("libmyModel.so");   // custom libraries, if any
```

## Running

From the master case, in a persistent session:

```bash
tmux new -s enkf
bash submit_and_update.sh
```

**Setup** (only when `iterationCount` is 1): detects the domains, creates `simK/EnKFmembers/member_i/`, decomposes the members if parallel, sets `endTime` and `writeInterval` in their `controlDict`, perturbs the parameters (identically in all domains), writes the perturbed observations and pseudo-observations to `simK/constant/enkfPerturbations`, and generates `run_all_members.sh` and `run_member.sh`.

**Each cycle** *c* = `iterationCount` … `nCycles`:

1. `endTime` is set to *c* × `updateInterval[k]` in every member of domain *k*;
2. all members run in one SLURM job; each then runs `reconstructPar -latestTime` and `postProcess -func sampleDict -latestTime`, and older time directories are deleted;
3. the driver checks that every member reached its checkpoint and was sampled;
4. the update binary computes the analysis and rewrites the parameter file of every member;
5. `iterationCount` is increased in the master `EnKFDict`.

Every job's final state is checked with `sacct`; any state other than `COMPLETED` stops the driver.

**Resuming:** set `iterationCount` to the next cycle and start the driver again. Setup is skipped when `iterationCount > 1`.

**Running by hand:**

```bash
EnKF_FoamMultiVariable_multiDomain_clustering_v10 > log.setup         # iterationCount 1
sbatch run_all_members.sh                                            # wait until finished
EnKF_FoamMultiVariable_multiDomain_clustering_Update_v10 > log.update
# then increase iterationCount, set endTime in every member, and repeat
```

Before a fresh restart, delete `simK/EnKFmembers/` and the generated scripts.

## Outputs

| Path | Content |
|---|---|
| `simK/EnKFmembers/member_i/` | one case per member; its parameter file is rewritten every cycle |
| `simK/constant/enkfPerturbations` | perturbed observations and pseudo-observations, reused in every cycle |
| `member_i/postProcessing/sampleDict/<t>/<set>.xy` | sampled values at the observation points |
| `log.<update>_cycleN` | full analysis log of cycle *N* |

The update log reports, per cycle: the residuals against the observations per domain and field; the localisation weights per cluster (useful to tune $L$); the Kalman-gain ranges; and, per parameter and cluster, the contribution of each domain and field, the most influential observations, the resamplings at the bounds, and the final ensemble mean, standard deviation and range.

## Assumptions and limitations

* Steady or pseudo-time solvers only: time = iteration count, checkpoints at *c* × `updateInterval`.
* A field named `U` must exist in every domain.
* If a member stops before its checkpoint (e.g. through `residualControl`), its latest time directory is used as the checkpoint.
* Parameter defaults must be strictly positive.
* One component per field, and one absolute `obsUncertainty` for all observations: fields of very different magnitude receive the same error.
* The localisation distance is absolute: an observation far from every cluster informs no parameter, and $L$ must be retuned when the clustering changes.
* The analysis memory grows with the case size, since all cells of all assimilated fields are stored for every member and domain.
* SLURM only (`srun --exact` requires SLURM ≥ 20.11); the update application is serial; paths must not contain spaces.
* Random seeds come from the clock, so runs are not bit-reproducible.
