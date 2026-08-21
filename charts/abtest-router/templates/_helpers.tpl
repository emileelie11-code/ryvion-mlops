{{/*
The in-cluster address of one colour of charts/abtest-model.

Called with a dict carrying the root context and a colour, because a named
template has no access to `.Values` unless it is handed it:

    include "abtest-router.url" (dict "root" $ "colour" "blue")
*/}}
{{- define "abtest-router.url" -}}
{{- $u := .root.Values.upstream -}}
{{- printf "http://%s-%s.%s.svc.cluster.local:%d" $u.service .colour .root.Values.namespace (int $u.port) -}}
{{- end -}}

{{/*
The routing table, as the JSON the router reads from ROUTER_ROUTES.

Every colour in this list may answer a caller. Note what is NOT here in shadow
mode: the mirror target. The two are separate on purpose - a mirror that appears
in the routing table is not a shadow deployment, it is a canary with extra steps.
*/}}
{{- define "abtest-router.routes" -}}
{{- $routes := list -}}
{{- if eq .Values.strategy "bluegreen" -}}
  {{- $url := include "abtest-router.url" (dict "root" . "colour" .Values.live) -}}
  {{- $routes = append $routes (dict "name" .Values.live "url" $url "weight" 100) -}}
{{- else if eq .Values.strategy "canary" -}}
  {{- $weight := int .Values.canary.weight -}}
  {{- if or (lt $weight 0) (gt $weight 100) -}}
    {{- fail (printf "canary.weight must be between 0 and 100, got %d" $weight) -}}
  {{- end -}}
  {{- $stable := include "abtest-router.url" (dict "root" . "colour" .Values.canary.stable) -}}
  {{- $candidate := include "abtest-router.url" (dict "root" . "colour" .Values.canary.candidate) -}}
  {{- $routes = append $routes (dict "name" .Values.canary.stable "url" $stable "weight" (sub 100 $weight)) -}}
  {{- $routes = append $routes (dict "name" .Values.canary.candidate "url" $candidate "weight" $weight) -}}
{{- else if eq .Values.strategy "shadow" -}}
  {{- $url := include "abtest-router.url" (dict "root" . "colour" .Values.shadow.serving) -}}
  {{- $routes = append $routes (dict "name" .Values.shadow.serving "url" $url "weight" 100) -}}
{{- else -}}
  {{- fail (printf "strategy must be bluegreen, canary or shadow, got %q" .Values.strategy) -}}
{{- end -}}
{{- $routes | toJson -}}
{{- end -}}

{{/*
The mirror target, as ROUTER_MIRROR - and the empty string in every strategy but
shadow, which is how the router is told to mirror nothing.
*/}}
{{- define "abtest-router.mirror" -}}
{{- if eq .Values.strategy "shadow" -}}
  {{- if eq (toString .Values.shadow.mirror) (toString .Values.shadow.serving) -}}
    {{- fail "shadow.mirror and shadow.serving must be different colours" -}}
  {{- end -}}
  {{- $url := include "abtest-router.url" (dict "root" . "colour" .Values.shadow.mirror) -}}
  {{- dict "name" .Values.shadow.mirror "url" $url "percentage" (int .Values.shadow.percentage) | toJson -}}
{{- end -}}
{{- end -}}
