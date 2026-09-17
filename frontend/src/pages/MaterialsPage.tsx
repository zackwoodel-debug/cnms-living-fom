import { useState } from "react";
import { Link } from "react-router-dom";

import { Async } from "../components/Async";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";

export default function MaterialsPage() {
  const [formula, setFormula] = useState("");
  const [query, setQuery] = useState("");
  const materials = useAsync(() => api.listMaterials({ formula: query }), [query]);

  return (
    <>
      <div className="panel">
        <h2>Materials</h2>
        <p className="hint">
          One row is one material-context record: composition, polymorph, and specimen form
          together. A chemical formula alone is not a material identifier &mdash; &ldquo;TiO<sub>2</sub>&rdquo;
          may be rutile, anatase, brookite, amorphous, a doped film, or a ceramic, and those do not
          share properties.
        </p>
        <form
          className="row"
          onSubmit={(event) => {
            event.preventDefault();
            setQuery(formula.trim());
          }}
        >
          <input
            placeholder="Filter by formula, e.g. HfO2"
            value={formula}
            onChange={(event) => setFormula(event.target.value)}
          />
          <button type="submit">Search</button>
          {query && (
            <button
              type="button"
              onClick={() => {
                setFormula("");
                setQuery("");
              }}
            >
              Clear
            </button>
          )}
        </form>
      </div>

      <Async
        state={materials}
        isEmpty={(rows) => rows.length === 0}
        empty="No materials yet. POST to /materials, or run `cnms-fom seed` for the draft setup."
      >
        {(rows) => (
          <div className="panel">
            <table>
              <thead>
                <tr>
                  <th>Formula</th>
                  <th>Polymorph</th>
                  <th>Specimen form</th>
                  <th>Space group</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((material) => (
                  <tr key={material.id}>
                    <td>
                      <Link to={`/materials/${material.id}`}>{material.formula_reduced}</Link>
                    </td>
                    <td>{material.polymorph}</td>
                    <td className="muted">{material.specimen_form.replace(/_/g, " ")}</td>
                    <td className="mono muted">{material.space_group_symbol ?? "—"}</td>
                    <td className="muted">
                      {material.source_database ? material.source_database : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Async>
    </>
  );
}
