import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({
      to: "/chat/$threadId",
      params: { threadId: "hf02-screening" },
    });
  },
  head: () => ({
    meta: [
      { title: "Quark · CNMS Research Agent" },
      {
        name: "description",
        content: "A local, evidence-first materials research workspace for CNMS Living FOM.",
      },
      { property: "og:title", content: "Quark · CNMS Research Agent" },
      {
        property: "og:description",
        content: "A local, evidence-first materials research workspace for CNMS Living FOM.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
});
