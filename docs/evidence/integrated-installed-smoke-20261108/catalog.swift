import Foundation
@main struct CatalogCheck {
    static func main() throws {
        let rows = numiWindowPolicyCatalog().map { choice in
            ["path":choice.policyURL.path, "role":choice.evidenceStatus.role,
             "balance":choice.evidenceStatus.balance.rawValue]
        }
        let data = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
        guard rows.contains(where: { $0["role"] == "selected incumbent" || $0["role"] == "selected candidate" }),
              rows.contains(where: { $0["role"] == "candidate experiment" }),
              rows.allSatisfy({ $0["balance"] != "guided held-out balance status unknown" }) else {
            throw NSError(domain:"catalog-check",code:1)
        }
    }
}
