from reference.rari_neurobrain.subject import Brain,Parcel
class LocalFleet:
    def __init__(self):self.states=Brain().checkpoint();self.last_events=[]
    def restore(self,state):Brain.restore(state);self.states=state
    def checkpoint(self):Brain.restore(self.states);return self.states
    def block(self,channels,perturbations,*,operation_id,reset_each=False):
        if len(channels)!=len(perturbations) or not channels:raise ValueError('BLOCK_SHAPE')
        before=self.states;brain=Brain.restore(before);tract=[p.observe()['action'] for p in brain.parcels]
        parcelobs=[];states=[];events=[]
        for i in range(101):
            p=Parcel.restore(before[i]);obs=[]
            for x,pt in zip(channels,perturbations):
                if reset_each:p=Parcel(i+1)
                obs.append(p.step([x[j%2] for j in range(p.n)],0.0 if reset_each else tract[(i-1)%101],pt));events.extend(p.events)
            parcelobs.append(obs);states.append(p.checkpoint())
        self.states=states;self.last_events=events
        return [[parcelobs[i][t] for i in range(101)] for t in range(len(channels))]
