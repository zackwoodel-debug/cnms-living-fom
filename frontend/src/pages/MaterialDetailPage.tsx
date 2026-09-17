import { Link, useParams } from "react-router-dom";

import { Async } from "../components/Async";
import { TierBadge } from "../components/Badges";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";

const fmt = (value: number | null, digits = 4) =>
  value === null ? "NA" : Number(value).toPrecision(digits);

export default function MaterialDetailPage() {
  const { id } = useParams<{ id: string }>();
  const material = useAsync(() => api.getMaterial(Number(id)), [id]);

  return (
    <Async state={material}>
      {(data) => (
        <>
          <div className="panel">
            <h2>
              {data.formula_reduced} <span className="muted">/ {data.polymorph}</span>
            </h2>
            <p className="hint">
              {data.specimen_form.replace(/_/g, " ")}
              {data.space_group_symbol ? ` · ${data.space_group_symbol}` : ""}
              {data.source_database ? ` · ${data.source_database} ${data.source_identifier ?? ""}` : ""}
            </p>
            <Link to="/materials">&larr; All materials</Link>
          </div>

          <div className="panel">
            <h2>Structural descriptors (S)</h2>
            <p className="hint">
              Z*<sub>RMS</sub>, &omega;<sub>TO,min</sub>, S<sub>osc</sub>, and A<sub>&epsilon;</sub>
              are not derivable from geometry &mdash; they come from DFPT or spectroscopy and carry
              their own method metadata.
            </p>
            {data.descriptors.length === 0 ? (
              <p className="muted">
                None computed. POST a CIF to <code>/materials/{data.id}/structure</code>, then{" "}
                <code>/materials/{data.id}/descriptors/compute</code>.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Descriptor</th>
                    <th className="num">Value</th>
                    <th>Units</th>
                    <th>Method</th>
                    <th>Provenance</th>
                  </tr>
                </thead>
                <tbody>
                  {data.descriptors.map((descriptor) => (
                    <tr key={`${descriptor.descriptor_key}-${descriptor.method}`}>
                      <td className="mono">{descriptor.descriptor_key}</td>
                      <td className="num">{fmt(descriptor.value)}</td>
                      <td className="muted">{descriptor.units ?? "—"}</td>
                      <td className="muted">{descriptor.method ?? "—"}</td>
                      <td>
                        <TierBadge tier={descriptor.provenance_tier} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel">
            <h2>Physical properties (P)</h2>
            <p className="hint">
              Each value carries its own measurement context. Two dielectric constants at different
              frequencies are two rows, never an average.
            </p>
            {data.properties.length === 0 ? (
              <p className="muted">No properties recorded.</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Property</th>
                    <th className="num">Value</th>
                    <th>Units</th>
                    <th>Component</th>
                    <th className="num">T (K)</th>
                    <th className="num">f (Hz)</th>
                    <th>Method</th>
                    <th>Provenance</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {data.properties.map((property) => (
                    <tr key={property.id}>
                      <td className="mono">{property.property_key}</td>
                      <td className="num">{fmt(property.value)}</td>
                      <td className="muted">{property.units ?? "—"}</td>
                      <td className="muted">{property.tensor_component ?? "—"}</td>
                      <td className="num">{property.temperature_k ?? "—"}</td>
                      <td className="num">
                        {property.frequency_hz ? property.frequency_hz.toExponential(1) : "—"}
                      </td>
                      <td className="muted">{property.method ?? "—"}</td>
                      <td>
                        <TierBadge tier={property.provenance_tier} />
                      </td>
                      <td className="muted">
                        {property.doi ? (
                          <a href={`https://doi.org/${property.doi}`} target="_blank" rel="noreferrer">
                            doi
                          </a>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </Async>
  );
}
