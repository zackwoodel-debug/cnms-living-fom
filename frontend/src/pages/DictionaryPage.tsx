import { Async } from "../components/Async";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";

export default function DictionaryPage() {
  const dictionary = useAsync(() => api.dictionary(), []);

  return (
    <Async state={dictionary}>
      {(data) => (
        <>
          <div className="panel">
            <h2>Descriptor dictionary</h2>
            <p className="hint">
              Ships with every released analysis: formula, units, source method, physical
              interpretation, declared transform, and missing-value policy. It is generated from the
              same registry the engine computes against, so it cannot drift from what was actually
              calculated.
            </p>
          </div>

          <div className="panel">
            <h2>Structural descriptors (S)</h2>
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Name</th>
                  <th>Units</th>
                  <th>Formula</th>
                  <th>Transform</th>
                  <th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {data.structural_descriptors.map((descriptor) => (
                  <tr key={descriptor.key}>
                    <td className="mono">{descriptor.key}</td>
                    <td>{descriptor.name}</td>
                    <td className="muted">{descriptor.units}</td>
                    <td className="mono muted">{descriptor.formula}</td>
                    <td className="muted">{descriptor.default_transform}</td>
                    <td className="muted">
                      {descriptor.interpretation}
                      {descriptor.caveat && (
                        <div style={{ color: "var(--warn)", fontSize: 12, marginTop: 4 }}>
                          {descriptor.caveat}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <h2>Physical properties (P)</h2>
            <p className="hint">
              &ldquo;Required context&rdquo; is enforced on write: a value missing any of these
              fields is rejected, because it cannot be compared with any other value.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Name</th>
                  <th>Units</th>
                  <th>Direction</th>
                  <th>Required context</th>
                </tr>
              </thead>
              <tbody>
                {data.physical_properties.map((property) => (
                  <tr key={property.key}>
                    <td className="mono">{property.key}</td>
                    <td>{property.name}</td>
                    <td className="muted">{property.units}</td>
                    <td>
                      <span className={`badge ${property.direction === "benefit" ? "ok" : "warn"}`}>
                        {property.direction}
                      </span>
                    </td>
                    <td className="mono muted">
                      {property.required_context.length
                        ? property.required_context.join(", ")
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Async>
  );
}
