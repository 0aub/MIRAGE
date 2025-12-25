import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  Loader2,
  Settings as SettingsIcon,
  Server,
  Cpu,
  Database,
  Blocks,
  CheckCircle2,
  XCircle,
  FileText,
  Zap,
  Globe,
  Lock,
  Layers,
  Box,
  Eye,
  EyeOff
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { dbApi } from "@/lib/api";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface SystemSettings {
  models?: any;
  processing?: any;
  databases?: any;
  chunking_strategies?: any;
  prompts?: Record<string, string>;
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<SystemSettings>({});
  const [isLoading, setIsLoading] = useState(true);
  const [configDir, setConfigDir] = useState("");
  const [showPasswords, setShowPasswords] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setIsLoading(true);
      const response = await dbApi.getSettings();
      setSettings(response.settings);
      setConfigDir(response.config_dir);
    } catch (error: any) {
      console.error('Error loading settings:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to load system settings",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const renderBooleanValue = (value: boolean) => (
    <div className="flex items-center gap-2">
      {value ? (
        <>
          <CheckCircle2 className="w-4 h-4 text-green-500" />
          <span className="text-green-600 dark:text-green-400 font-medium">Enabled</span>
        </>
      ) : (
        <>
          <XCircle className="w-4 h-4 text-red-500" />
          <span className="text-red-600 dark:text-red-400 font-medium">Disabled</span>
        </>
      )}
    </div>
  );

  const renderModelProviders = (llmConfig: any) => {
    if (!llmConfig?.providers) return null;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {Object.entries(llmConfig.providers).map(([provider, config]: [string, any]) => {
          // Get token constraints for this provider
          const constraints = llmConfig.constraints?.[provider];

          return (
            <Card key={provider} className="border-2 hover:border-primary/50 transition-colors">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 bg-primary/10 rounded-lg">
                      <Cpu className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="capitalize text-base">{provider}</CardTitle>
                      <CardDescription className="text-xs mt-0.5">
                        {config.description || "LLM Provider"}
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex flex-col gap-1 items-end">
                    {config.enabled ? (
                      <Badge className="bg-green-600 hover:bg-green-700 text-xs">Active</Badge>
                    ) : (
                      <Badge variant="secondary" className="text-xs">Inactive</Badge>
                    )}
                    <Badge variant="outline" className="text-xs">
                      Priority: {config.priority}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Model Configuration Table */}
                <div className="rounded-lg border overflow-hidden">
                  <Table>
                    <TableBody>
                      {config.model && (
                        <TableRow>
                          <TableCell className="font-medium text-muted-foreground w-1/3">Model</TableCell>
                          <TableCell className="font-mono text-sm">{config.model}</TableCell>
                        </TableRow>
                      )}
                      {config.endpoint && (
                        <TableRow>
                          <TableCell className="font-medium text-muted-foreground w-1/3">Endpoint</TableCell>
                          <TableCell className="font-mono text-xs break-all">{config.endpoint}</TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>

                {/* Token Constraints - Inline */}
                {constraints && (
                  <div className="bg-secondary/30 rounded-lg p-3 space-y-2">
                    <div className="flex items-center gap-2 mb-1">
                      <Zap className="w-3.5 h-3.5 text-amber-500" />
                      <h4 className="font-semibold text-xs">Token Constraints</h4>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {constraints.max_input_tokens && (
                        <div className="space-y-1">
                          <p className="text-xs text-muted-foreground">Max Input</p>
                          <Badge variant="secondary" className="font-mono text-xs">
                            {constraints.max_input_tokens.toLocaleString()}
                          </Badge>
                        </div>
                      )}
                      {constraints.max_output_tokens && (
                        <div className="space-y-1">
                          <p className="text-xs text-muted-foreground">Max Output</p>
                          <Badge variant="secondary" className="font-mono text-xs">
                            {constraints.max_output_tokens.toLocaleString()}
                          </Badge>
                        </div>
                      )}
                      {constraints.context_window && (
                        <div className="space-y-1">
                          <p className="text-xs text-muted-foreground">Context Window</p>
                          <Badge variant="secondary" className="font-mono text-xs">
                            {constraints.context_window.toLocaleString()}
                          </Badge>
                        </div>
                      )}
                      {constraints.supports_streaming !== undefined && (
                        <div className="space-y-1">
                          <p className="text-xs text-muted-foreground">Streaming</p>
                          {renderBooleanValue(constraints.supports_streaming)}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    );
  };

  const renderLLMParameters = (parameters: any) => {
    if (!parameters) return null;

    return (
      <Card className="border-2">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-indigo-500/10 rounded-lg">
              <Layers className="w-4 h-4 text-indigo-500" />
            </div>
            <div>
              <CardTitle className="text-base">LLM Parameters</CardTitle>
              <CardDescription className="text-xs">Generation settings</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(parameters).map(([key, value]) => (
              <div key={key} className="bg-secondary/30 rounded-lg p-3 space-y-1.5">
                <p className="text-xs text-muted-foreground capitalize">
                  {key.replace(/_/g, ' ')}
                </p>
                <div className="flex items-center">
                  {typeof value === 'boolean' ? (
                    renderBooleanValue(value)
                  ) : typeof value === 'number' ? (
                    <Badge variant="secondary" className="font-mono text-sm">{value}</Badge>
                  ) : (
                    <span className="font-mono text-sm">{String(value)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderEmbeddings = (embeddings: any) => {
    if (!embeddings) return null;

    // Filter Jina embeddings to only show model and embedding_dim
    const jinaConfig = embeddings.jina ? {
      model: embeddings.jina.model,
      embedding_dim: embeddings.jina.embedding_dim
    } : null;

    return (
      <div className="space-y-4">
        <Card className="border-2">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-purple-500/10 rounded-lg">
                <Globe className="w-4 h-4 text-purple-500" />
              </div>
              <div>
                <CardTitle className="text-base">Jina Embeddings</CardTitle>
                <CardDescription className="text-xs">Text embedding configuration</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            {jinaConfig ? (
              <div className="rounded-lg border overflow-hidden">
                <Table>
                  <TableBody>
                    {jinaConfig.model && (
                      <TableRow>
                        <TableCell className="font-medium text-muted-foreground w-1/3">Model</TableCell>
                        <TableCell className="font-mono text-sm">{jinaConfig.model}</TableCell>
                      </TableRow>
                    )}
                    {jinaConfig.embedding_dim && (
                      <TableRow>
                        <TableCell className="font-medium text-muted-foreground w-1/3">Embedding Dimension</TableCell>
                        <TableCell>
                          <Badge variant="secondary" className="font-mono">{jinaConfig.embedding_dim}</Badge>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No Jina configuration available</p>
            )}
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderProcessingSection = (sectionName: string, config: any) => {
    if (!config || typeof config !== 'object') return null;

    // Get icon based on section name
    const getIcon = () => {
      const name = sectionName.toLowerCase();
      if (name.includes('extraction')) return <Box className="w-4 h-4 text-blue-500" />;
      if (name.includes('rewriting')) return <FileText className="w-4 h-4 text-green-500" />;
      if (name.includes('compression')) return <Layers className="w-4 h-4 text-purple-500" />;
      return <Box className="w-4 h-4 text-orange-500" />;
    };

    return (
      <Card key={sectionName} className="border-2">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-primary/10 rounded-lg">
              {getIcon()}
            </div>
            <div>
              <CardTitle className="capitalize text-base">
                {sectionName.replace(/_/g, ' ')}
              </CardTitle>
              <CardDescription className="text-xs">Configuration</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(config).map(([key, value]) => {
              if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                return null; // Skip nested objects for cleaner display
              }
              return (
                <div key={key} className="bg-secondary/30 rounded-lg p-3 space-y-1.5">
                  <p className="text-xs text-muted-foreground capitalize">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <div className="flex items-center">
                    {typeof value === 'boolean' ? (
                      renderBooleanValue(value)
                    ) : typeof value === 'number' ? (
                      <Badge variant="secondary" className="font-mono text-sm">{value}</Badge>
                    ) : Array.isArray(value) ? (
                      <div className="flex flex-wrap gap-1">
                        {value.map((item, idx) => (
                          <Badge key={idx} variant="outline" className="font-mono text-xs">
                            {String(item)}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <span className="font-mono text-sm break-all">{String(value)}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderDatabases = () => {
    if (!settings.databases) return null;

    // Filter out MongoDB
    const filteredDatabases = Object.entries(settings.databases).filter(
      ([db]) => db.toLowerCase() !== 'mongodb'
    );

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredDatabases.map(([db, config]: [string, any]) => (
          <Card key={db} className="border-2 hover:border-primary/50 transition-colors">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-blue-500/10 rounded-lg">
                  <Database className="w-4 h-4 text-blue-500" />
                </div>
                <div>
                  <CardTitle className="uppercase text-base">{db}</CardTitle>
                  <CardDescription className="text-xs">{config.description || "Database connection"}</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="rounded-lg border overflow-hidden">
                <Table>
                  <TableBody>
                    {Object.entries(config).map(([key, value]) => {
                      // Skip nested objects and description
                      if (key === 'description' || (typeof value === 'object' && value !== null && !Array.isArray(value))) {
                        return null;
                      }

                      // Check if this is a password/secret field
                      const isPasswordField = key.toLowerCase().includes('password') || key.toLowerCase().includes('secret');

                      // Mask sensitive data unless showPasswords is true
                      let displayValue = value;
                      if (isPasswordField && !showPasswords) {
                        displayValue = '••••••••';
                      }

                      return (
                        <TableRow key={key}>
                          <TableCell className="font-medium text-muted-foreground w-1/3 capitalize">
                            {key.replace(/_/g, ' ')}
                          </TableCell>
                          <TableCell>
                            {typeof value === 'boolean' ? (
                              renderBooleanValue(value)
                            ) : typeof value === 'number' ? (
                              <Badge variant="secondary" className="font-mono">{value}</Badge>
                            ) : isPasswordField ? (
                              <div className="flex items-center gap-2">
                                <Lock className="w-3 h-3 text-muted-foreground" />
                                <span className="font-mono text-sm">{String(displayValue)}</span>
                              </div>
                            ) : (
                              <span className="font-mono text-sm break-all">{String(value)}</span>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  const renderChunkingStrategies = () => {
    if (!settings.chunking_strategies) return null;

    return (
      <Accordion type="multiple" className="space-y-2">
        {Object.entries(settings.chunking_strategies).map(([strategy, config]: [string, any]) => (
          <AccordionItem key={strategy} value={strategy} className="border-2 rounded-lg">
            <AccordionTrigger className="px-6 py-4 hover:no-underline">
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-3">
                  <Blocks className="w-5 h-5 text-orange-500" />
                  <span className="font-semibold capitalize">{strategy.replace(/_/g, ' ')}</span>
                </div>
                {config.strategy && (
                  <Badge variant="outline" className="capitalize ml-4">
                    {config.strategy}
                  </Badge>
                )}
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pb-4">
              <div className="rounded-lg border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-1/3">Parameter</TableHead>
                      <TableHead>Value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(config).map(([key, value]) => {
                      // Skip description and nested objects
                      if (key === 'description' || (typeof value === 'object' && value !== null && !Array.isArray(value))) {
                        return null;
                      }

                      return (
                        <TableRow key={key}>
                          <TableCell className="font-medium capitalize">
                            {key.replace(/_/g, ' ')}
                          </TableCell>
                          <TableCell>
                            {typeof value === 'boolean' ? (
                              renderBooleanValue(value)
                            ) : typeof value === 'number' ? (
                              <Badge variant="secondary" className="font-mono">{value}</Badge>
                            ) : Array.isArray(value) ? (
                              <div className="flex flex-wrap gap-1">
                                {value.map((item, idx) => (
                                  <Badge key={idx} variant="outline" className="font-mono text-xs">
                                    {String(item)}
                                  </Badge>
                                ))}
                              </div>
                            ) : (
                              <span className="font-mono text-sm">{String(value)}</span>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    );
  };

  const renderPrompts = () => {
    if (!settings.prompts || Object.keys(settings.prompts).length === 0) {
      return (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileText className="w-12 h-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">No prompts available</p>
          </CardContent>
        </Card>
      );
    }

    // Separate built-in and file-based prompts
    const builtinPrompts: [string, string][] = [];
    const filePrompts: [string, string][] = [];

    Object.entries(settings.prompts).forEach(([name, content]) => {
      if (name.startsWith('builtin:')) {
        builtinPrompts.push([name.replace('builtin:', ''), content]);
      } else if (name.startsWith('file:')) {
        filePrompts.push([name.replace('file:', ''), content]);
      } else {
        // Legacy format without prefix
        filePrompts.push([name, content]);
      }
    });

    const renderPromptCard = (name: string, content: string, isBuiltin: boolean) => (
      <Card key={name} className="border-2">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`p-1.5 rounded-lg ${isBuiltin ? 'bg-purple-500/10' : 'bg-green-500/10'}`}>
                <FileText className={`w-4 h-4 ${isBuiltin ? 'text-purple-500' : 'text-green-500'}`} />
              </div>
              <div>
                <CardTitle className="text-base">
                  {name.replace(/_/g, ' ').replace('.txt', '')}
                </CardTitle>
                <CardDescription className="text-xs">
                  {isBuiltin ? 'Built-in prompt template' : 'File-based prompt template'}
                </CardDescription>
              </div>
            </div>
            <Badge variant={isBuiltin ? 'default' : 'secondary'} className="text-xs">
              {isBuiltin ? 'Built-in' : 'Custom'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="bg-secondary/30 rounded-lg p-4 font-mono text-sm whitespace-pre-wrap max-h-96 overflow-y-auto">
            {content}
          </div>
        </CardContent>
      </Card>
    );

    return (
      <div className="space-y-6">
        {builtinPrompts.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Box className="w-5 h-5 text-purple-500" />
              <h3 className="font-semibold">Built-in Prompts ({builtinPrompts.length})</h3>
            </div>
            <div className="space-y-4">
              {builtinPrompts.map(([name, content]) => renderPromptCard(name, content, true))}
            </div>
          </div>
        )}

        {filePrompts.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold">Custom File Prompts ({filePrompts.length})</h3>
            </div>
            <div className="space-y-4">
              {filePrompts.map(([name, content]) => renderPromptCard(name, content, false))}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin mb-4" />
          <p className="text-muted-foreground">Loading configurations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-in pb-20 md:pb-0">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-1">Configurations</h1>
          <p className="text-muted-foreground text-sm">System configuration and settings</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPasswords(!showPasswords)}
            className="flex items-center gap-2"
          >
            {showPasswords ? (
              <>
                <EyeOff className="w-4 h-4" />
                <span className="hidden md:inline">Hide Passwords</span>
              </>
            ) : (
              <>
                <Eye className="w-4 h-4" />
                <span className="hidden md:inline">Show Passwords</span>
              </>
            )}
          </Button>
          <SettingsIcon className="w-8 h-8 text-muted-foreground" />
        </div>
      </div>

      {configDir && (
        <Card className="bg-secondary/30 border-2">
          <CardContent className="pt-3 pb-3">
            <div className="flex items-center gap-2 text-sm">
              <Server className="w-4 h-4 text-muted-foreground" />
              <span className="text-muted-foreground">Configuration directory:</span>
              <code className="font-mono text-xs bg-secondary px-2 py-1 rounded">{configDir}</code>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Settings Tabs */}
      <Tabs defaultValue="models" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-5 h-auto">
          <TabsTrigger value="models" className="gap-2 py-3">
            <Cpu className="w-4 h-4" />
            <span className="hidden sm:inline">Models</span>
          </TabsTrigger>
          <TabsTrigger value="processing" className="gap-2 py-3">
            <Blocks className="w-4 h-4" />
            <span className="hidden sm:inline">Processing</span>
          </TabsTrigger>
          <TabsTrigger value="databases" className="gap-2 py-3">
            <Database className="w-4 h-4" />
            <span className="hidden sm:inline">Databases</span>
          </TabsTrigger>
          <TabsTrigger value="chunking" className="gap-2 py-3">
            <Server className="w-4 h-4" />
            <span className="hidden sm:inline">Chunking</span>
          </TabsTrigger>
          <TabsTrigger value="prompts" className="gap-2 py-3">
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Prompts</span>
          </TabsTrigger>
        </TabsList>

        {/* Models Tab */}
        <TabsContent value="models" className="space-y-4">
          {settings.models?.llm && (
            <div className="space-y-4">
              <div>
                <h2 className="text-xl font-semibold mb-1">LLM Providers</h2>
                <p className="text-muted-foreground text-sm mb-3">
                  Language model configurations and token constraints
                </p>
                {renderModelProviders(settings.models.llm)}
              </div>

              {settings.models.llm.parameters && (
                <div className="pt-4 border-t">
                  {renderLLMParameters(settings.models.llm.parameters)}
                </div>
              )}
            </div>
          )}

          {settings.models?.embeddings && (
            <div className="pt-4 border-t">
              <h2 className="text-xl font-semibold mb-1">Embeddings</h2>
              <p className="text-muted-foreground text-sm mb-3">
                Text embedding model configuration
              </p>
              {renderEmbeddings(settings.models.embeddings)}
            </div>
          )}
        </TabsContent>

        {/* Processing Tab */}
        <TabsContent value="processing" className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold mb-1">Processing Configuration</h2>
            <p className="text-muted-foreground text-sm mb-3">
              Document processing and analysis settings
            </p>
          </div>
          {settings.processing && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {Object.entries(settings.processing).map(([section, config]) =>
                renderProcessingSection(section, config)
              )}
            </div>
          )}
        </TabsContent>

        {/* Databases Tab */}
        <TabsContent value="databases" className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold mb-1">Database Connections</h2>
            <p className="text-muted-foreground text-sm mb-3">
              Active database configurations and connection details
            </p>
          </div>
          {renderDatabases()}
        </TabsContent>

        {/* Chunking Tab */}
        <TabsContent value="chunking" className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold mb-1">Chunking Strategies</h2>
            <p className="text-muted-foreground text-sm mb-3">
              Document chunking configurations for different content types
            </p>
          </div>
          {renderChunkingStrategies()}
        </TabsContent>

        {/* Prompts Tab */}
        <TabsContent value="prompts" className="space-y-4">
          <div>
            <h2 className="text-xl font-semibold mb-1">System Prompts</h2>
            <p className="text-muted-foreground text-sm mb-3">
              Prompt templates used across the system
            </p>
          </div>
          {renderPrompts()}
        </TabsContent>
      </Tabs>
    </div>
  );
}
