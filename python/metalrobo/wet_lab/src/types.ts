// Typed views of owner responses. Optional scientific extensions remain owner data.
export type OwnerData = Record<string, any>;
export interface Named {id:string; title?:string; specimenID?:string; conditionID?:string; cellType?:string; controlCells?:number}
export interface Selection {assayID:string; specimenID:string; populationID:string; conditionID:string; gene?:string; target?:string; experimentID?:string|null}
export interface Capability {presentation:string; prediction:boolean; measuredExploration?:boolean; predictionReason?:string; features?:string[]; conditions:Named[]; populations:Named[]; specimens:Named[]}
export interface Catalog extends Capability {features:string[];featuresTotal?:number;datasetFingerprint?:string;observationAccessFingerprint?:string;modelVersion?:string;id:string; title:string; family:string; capabilities?:Capability; targets:OwnerData[]; limits?:OwnerData}
export interface Proposal {id:string;title:string;status:string;selection:Selection;genes:string[];targets:string[];modelIDs:string[];conditionIDs:string[];coverage:OwnerData;axes:OwnerData[];decisionSupport?:OwnerData;modelDifference?:OwnerData;record?:string;axisResults?:OwnerData[];result?:OwnerData;operation?:string;undo?:OwnerData[]}
export interface Operation {id:string;draftID:string;status:string;action:string;progress?:string;error?:string}
export interface SharedState {revision:number;selection:Selection|null;selectionSupport?:OwnerData;models:OwnerData[];datasets:OwnerData[];drafts:Proposal[];operations:Operation[];history:OwnerData[]}
export interface Readout extends OwnerData {gene:string;units:string;arms?:OwnerData[];access?:string|OwnerData;distributionStatus?:string}
export interface Evidence extends OwnerData {evidence?:string;source?:OwnerData;modelVersion?:string;biologicalPromotion?:boolean}
