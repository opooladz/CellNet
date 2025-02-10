import Foundation
import HCBacktrace
import Honeycrisp

public class OneSidedKron: Optimizer {
    public var lr: Float
    public var b1: Float
    public var weightDecay: Float
    public var preconditionerUpdateProbability: Float
    public var precondLr: Float
    public var clipUpdateRms: Bool
    
    public var moments: [String: Tensor] = [:]
    public var whiteningMatrices: [String: Tensor] = [:]
    
    public init(
        _ parameters: [(String, Trainable.Parameter)],
        lr: Float = 0.0003,
        b1: Float = 0.9,
        weightDecay: Float = 0.0,
        preconditionerUpdateProbability: Float = 1.0,
        precondLr: Float = 0.1,
        clipUpdateRms: Bool = true
    ) {
        self.lr = lr
        self.b1 = b1
        self.weightDecay = weightDecay
        self.preconditionerUpdateProbability = preconditionerUpdateProbability
        self.precondLr = precondLr
        self.clipUpdateRms = clipUpdateRms
        super.init(parameters)
    }

    @recordCaller private func _step() {
        for (name, var param) in parameters {
            guard var g = param.grad else { continue }
            var v = moments[name] ?? Tensor(zerosLike: g)
            v = v * b1 + g * (1 - b1)
            moments[name] = v
            g = v / (1 - pow(b1, Float(step)))
            
            let (m, n) = (g.shape[g.shape.count - 2], g.shape[g.shape.count - 1])
            let isTransposed = m < n
            if isTransposed { g = g.swap(axis: -2, with: -1) }
            
            var Q = whiteningMatrices[name] ?? Tensor(eye: g.shape[g.shape.count - 1])
            if Tensor.random(in: 0.0..<1.0).scalarized() < preconditionerUpdateProbability {
                Q = oneSidedPrecondUpdate(g, Q, precondLr: precondLr)
                whiteningMatrices[name] = Q
            }
            
            g = g.matmul(Q.transposed()).matmul(Q)
            if isTransposed { g = g.swap(axis: -2, with: -1) }
            g = g * pow(max(1.0, Float(m) / Float(n)), 0.5)
            
            if clipUpdateRms {
                g = clipUpdateRms(g)
            }
            
            param.data = param.data! - g * lr
        }
    }

    @recordCaller static private func oneSidedPrecondUpdate(_ G: Tensor, _ Q: Tensor, precondLr: Float) -> Tensor {
        let m = G.shape[G.shape.count - 2]
        let n = G.shape[G.shape.count - 1]
        if m < n { G = G.swap(axis: -2, with: -1) }
        
        let V = Tensor.random(shape: G.shape) / sqrt(Float(m))
        let Bh = safeSolveTriangular(Q, V, upper: true, left: false)
        let AhA = G.matmul(Q.transposed()).matmul(G)
        let BBh = Bh.transposed().matmul(Bh)
        return Q - (precondLr / normLowerBound(AhA + BBh)) * (AhA - BBh).triu().matmul(Q)
    }

    @recordCaller static private func normLowerBound(_ A: Tensor) -> Tensor {
        let maxAbs = A.norm(.inf)
        return maxAbs > 0 ? _lb(A, maxAbs) : maxAbs
    }

    @recordCaller static private func _lb(_ A: Tensor, _ maxAbs: Tensor) -> Tensor {
        let A = A / maxAbs
        let aa = A * A.conj()
        let value0 = aa.sum(axis: 0).max().scalarized()
        let value1 = aa.sum(axis: 1).max().scalarized()
        if value0 > value1 {
            let x = A[:, Tensor(argmax: aa.sum(axis: 0))].conj().matmul(A)
            return maxAbs * x.matmul(A.transposed()) / x.norm()
        } else {
            let x = A.matmul(A[Tensor(argmax: aa.sum(axis: 1))].conj())
            return maxAbs * A.transposed().matmul(x) / x.norm()
        }
    }

    @recordCaller static private func clipUpdateRms(_ g: Tensor) -> Tensor {
        let rms = g.square().mean().sqrt().add(1e-12)
        let scale = min(1.0, 1.1 / rms.scalarized())
        return g * scale
    }
}
